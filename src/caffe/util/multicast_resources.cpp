#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <time.h>
#include <limits.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>
#include <signal.h>
#include <pthread.h>

namespace caffe {

#ifdef RDMA
#include "caffe/util/multicast_resources.hpp"

// This is when we get sig handler from the user before we remove the join request.
struct mcast_parameters *sighandler_params;

/******************************************************************************
 * signalCatcher - cacth user signal in order to reregiser the mcast group
 ******************************************************************************/
static void signalCatcher (int sig)
{
	if (sig == SIGINT) {

		if (join_multicast_group(SUBN_ADM_METHOD_DELETE,sighandler_params))
			fprintf(stderr,"Couldn't Unregister the Mcast group on the SM\n");

		if (sighandler_params->is_2nd_mgid_used) {
			memcpy(sighandler_params->mgid.raw,sighandler_params->base_mgid.raw,16);
			if (join_multicast_group(SUBN_ADM_METHOD_DELETE,sighandler_params))
				fprintf(stderr,"Couldn't Unregister the Base Mcast group on the SM\n");
		}
	}
	exit(1);
}

/******************************************************************************
 * prepare_mcast_mad
 ******************************************************************************/
static void prepare_mcast_mad(uint8_t method,
							  struct mcast_parameters *params,
							  struct sa_mad_packet_t *samad_packet)	 {

	uint8_t *ptr;
	uint64_t comp_mask;

	memset(samad_packet,0,sizeof(*samad_packet));

	/* prepare the MAD header. according to Table 145 in IB spec 1.2.1 */
	ptr = samad_packet->mad_header_buf;
	ptr[0]                     = 0x01;					/* BaseVersion */
	ptr[1]                     = MANAGMENT_CLASS_SUBN_ADM;			/* MgmtClass */
	ptr[2]                     = 0x02; 					/* ClassVersion */
	ptr[3]                     = INSERTF(ptr[3], 0, method, 0, 7); 		/* Method */
	(*(uint64_t *)(ptr + 8))   = htonll((uint64_t)DEF_TRANS_ID);             /* TransactionID */
	(*(uint16_t *)(ptr + 16))  = htons(SUBN_ADM_ATTR_MC_MEMBER_RECORD);      /* AttributeID */

	ptr = samad_packet->SubnetAdminData;

	memcpy(&ptr[0],params->mgid.raw, 16);
	memcpy(&ptr[16],params->port_gid.raw, 16);

	(*(uint32_t *)(ptr + 32)) = htonl(DEF_QKEY);
	(*(uint16_t *)(ptr + 40)) = htons(params->pkey);
	ptr[39]                    = DEF_TCLASS;
	ptr[44]                    = INSERTF(ptr[44], 4, DEF_SLL, 0, 4);
	ptr[44]                    = INSERTF(ptr[44], 0, DEF_FLOW_LABLE, 16, 4);
	ptr[45]                    = INSERTF(ptr[45], 0, DEF_FLOW_LABLE, 8, 8);
	ptr[46]                    = INSERTF(ptr[46], 0, DEF_FLOW_LABLE, 0, 8);
	ptr[48]                    = INSERTF(ptr[48], 0, MCMEMBER_JOINSTATE_FULL_MEMBER, 0, 4);

	comp_mask = SUBN_ADM_COMPMASK_MGID | SUBN_ADM_COMPMASK_PORT_GID | SUBN_ADM_COMPMASK_Q_KEY |
				SUBN_ADM_COMPMASK_P_KEY | SUBN_ADM_COMPMASK_TCLASS | SUBN_ADM_COMPMASK_SL |
				SUBN_ADM_COMPMASK_FLOW_LABEL | SUBN_ADM_COMPMASK_JOIN_STATE;

	samad_packet->ComponentMask = htonll(comp_mask);
}

/******************************************************************************
 * check_mad_status
 ******************************************************************************/
static int check_mad_status(struct sa_mad_packet_t *samad_packet) {

	uint8_t *ptr;
	uint32_t user_trans_id;
	uint16_t mad_header_status;

	ptr = samad_packet->mad_header_buf;

	// the upper 32 bits of TransactionID were set by the kernel
    user_trans_id = ntohl(*(uint32_t *)(ptr + 12));

	// check the TransactionID to make sure this is the response
	// for the join/leave multicast group request we posted
	if (user_trans_id != DEF_TRANS_ID) {
		fprintf(stderr, "received a mad with TransactionID 0x%x, when expecting 0x%x\n",
			(unsigned int)user_trans_id, (unsigned int)DEF_TRANS_ID);;
		return 1;
	}

	mad_header_status = 0x0;
	mad_header_status = INSERTF(mad_header_status, 8, ptr[4], 0, 7);
	mad_header_status = INSERTF(mad_header_status, 0, ptr[5], 0, 8);

	if (mad_header_status) {
		fprintf(stderr,"received UMAD with an error: 0x%x\n", mad_header_status);
		return 1;
	}

	return 0;
}


/******************************************************************************
 * get_mlid_from_mad
 ******************************************************************************/
static void get_mlid_from_mad(struct sa_mad_packet_t *samad_packet,uint16_t *mlid) {
	uint8_t *ptr;
	ptr = samad_packet->SubnetAdminData;
	*mlid = ntohs(*(uint16_t *)(ptr + 36));
}

/******************************************************************************
 * get_gid_from_mad
 ******************************************************************************/
static void get_gid_from_mad(struct sa_mad_packet_t *samad_packet,ibv_gid *gid) {
  uint8_t *ptr;
  ptr = samad_packet->SubnetAdminData;
  memcpy(gid->raw, ptr, 16);
}

/******************************************************************************
 * set_multicast_gid
 ******************************************************************************/
void set_multicast_gid(struct mcast_parameters *params,uint32_t qp_num,int is_client) {

	uint8_t mcg_gid[16] = MCG_GID;
	char *pstr = const_cast<char*>(params->user_mgid);
	char *term = NULL;
	char tmp[20];
	int i;

	if (params->user_mgid) {
		term = strpbrk(pstr, ":");
		memcpy(tmp, pstr, term - pstr+1);
		tmp[term - pstr] = 0;

		mcg_gid[0] = (unsigned char)strtoll(tmp, NULL, 0);

		for (i = 1; i < 15; ++i) {
			pstr += term - pstr + 1;
			term = strpbrk(pstr, ":");
			memcpy(tmp, pstr, term - pstr+1);
			tmp[term - pstr] = 0;

			mcg_gid[i] = (unsigned char)strtoll(tmp, NULL, 0);
		}
		pstr += term - pstr + 1;

		strcpy(tmp, pstr);
		mcg_gid[15] = (unsigned char)strtoll(tmp, NULL, 0);
	}

	memcpy(params->mgid.raw,mcg_gid,16);
	if (is_client && params->user_mgid==NULL)
		params->mgid.raw[15]++;
}

/******************************************************************************
 * join_multicast_group
 ******************************************************************************/
int join_multicast_group(subn_adm_method method,struct mcast_parameters *params) {

	int portid = -1;
	int agentid = -1;
	void *umad_buff = NULL;
	void *mad = NULL;
	int length = MAD_SIZE;
	int test_result = 0;

	// mlid will be assigned to the new LID after the join
	if (umad_init() < 0) {
		fprintf(stderr, "failed to init the UMAD library\n");
		goto cleanup;
	}
	/* use casting to loose the "const char0 *" */
	portid = umad_open_port((char*)params->ib_devname,params->ib_port);
	if (portid < 0) {
		fprintf(stderr,"failed to open UMAD port %d\n",params->ib_port);
		goto cleanup;
	}

	agentid = umad_register(portid,MANAGMENT_CLASS_SUBN_ADM, 2, 0, 0);
	if (agentid < 0) {
		fprintf(stderr,"failed to register UMAD agent for MADs\n");
		goto cleanup;
	}

	umad_buff = umad_alloc(1, umad_size() + MAD_SIZE);
	if (!umad_buff) {
		fprintf(stderr, "failed to allocate MAD buffer\n");
		goto cleanup;
	}

	mad = umad_get_mad(umad_buff);
	prepare_mcast_mad(method,params,(struct sa_mad_packet_t *)mad);

	if (umad_set_addr(umad_buff,params->sm_lid,1,params->sm_sl,QP1_WELL_KNOWN_Q_KEY) < 0) {
		fprintf(stderr, "failed to set the destination address of the SMP\n");
		goto cleanup;
	}

	if (umad_send(portid,agentid,umad_buff,MAD_SIZE,100,5) < 0) {
		fprintf(stderr, "failed to send MAD\n");
		goto cleanup;
	}

	if (umad_recv(portid,umad_buff,&length,5000) < 0) {
		fprintf(stderr, "failed to receive MAD response\n");
		goto cleanup;
	}

	if (check_mad_status((struct sa_mad_packet_t*)mad)) {
		fprintf(stderr, "failed to get mlid from MAD\n");
		goto cleanup;
	}

	//  "Join multicast group" message was sent
	if (method == SUBN_ADM_METHOD_SET) {
	  get_gid_from_mad((struct sa_mad_packet_t*)mad,&params->mgid);
		get_mlid_from_mad((struct sa_mad_packet_t*)mad,&params->mlid);
		params->mcast_state |= MCAST_IS_JOINED;
		if (params->is_2nd_mgid_used == 0) {
			sighandler_params = params;
			signal(SIGINT,signalCatcher);
		}
	} else {
		params->mcast_state &= ~MCAST_IS_JOINED;
	}

cleanup:
	if (umad_buff)
		umad_free(umad_buff);

	if (portid >= 0) {
		if (agentid >= 0) {
			if (umad_unregister(portid, agentid)) {
				fprintf(stderr, "failed to deregister UMAD agent for MADs\n");
				test_result = 1;
			}
		}

		if (umad_close_port(portid)) {
			fprintf(stderr, "failed to close UMAD portid\n");
			test_result = 1;
		}
	}

	return test_result;
}

#endif /* RDMA */
}
