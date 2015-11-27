/*
 * Copyright (c) 2009 Mellanox Technologies Ltd.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Author: Ido Shamay <idos@dev.mellanox.co.il>
 */

#ifndef MULTICAST_RESOURCES_H
#define MULTICAST_RESOURCES_H

// Imported from perftest
#ifdef RDMA

 /* Multicast Module for perftest.
  *
  * Description :
  *
  *   This file contains the structures and methods for implementing a multiple
  *   multicast groups in user space enviroment.
  *	  The module is in use in "send_bw" and "send_lat" ,but can be used on other
  *	  applications and can generate more methods and serve more benchmarks.
  *   The Module uses only the structire defined here , enabling generic use of it.
  *
  * Defined Types :
  *
  *   mcast_parameters - Contains all the parameters needed for this module.
  *   mcast_group      - The multicast group entitiy itself.
  *   mcg_qp		   - Is a QP structure that is attahced to the group.
  *
  */


/************************************************************************
 *   Macros , Defines and Files included for work.	    			    *
 ************************************************************************/

#include <infiniband/verbs.h>
#include <infiniband/umad.h>
//#include "get_clock.h"

#define QPNUM_MCAST 				0xffffff
#define DEF_QKEY     			    0x11111111
#define DEF_PKEY_IDX        		0
#define DEF_SLL              		0
#define MAX_POLL_ITERATION_TIMEOUT  1000000
#define MCG_GID {255,1,0,0,0,2,201,133,0,0,0,0,0,0,0,0}

//  Definitions section for MADs
#define SUBN_ADM_ATTR_MC_MEMBER_RECORD 0x38
#define MANAGMENT_CLASS_SUBN_ADM       0x03 	  /* Subnet Administration class */
#define MCMEMBER_JOINSTATE_FULL_MEMBER 0x1
#define MAD_SIZE                       256	      /* The size of a MAD is 256 bytes */
#define QP1_WELL_KNOWN_Q_KEY           0x80010000 /* Q_Key value of QP1 */
#define DEF_TRANS_ID                   0x12345678 /* TransactionID */
#define DEF_TCLASS                     0
#define DEF_FLOW_LABLE                 0

// Macro for 64 bit variables to switch to from net
#ifndef ntohll
#define ntohll(x) (((uint64_t)(ntohl((int)((x << 32) >> 32))) << 32) | (unsigned int)ntohl(((int)(x >> 32))))
#endif
#ifndef htonll
#define htonll(x) ntohll(x)
#endif

// generate a bit mask S bits width
#define MASK32(S)  ( ((uint32_t) ~0L) >> (32-(S)) )

// generate a bit mask with bits O+S..O set (assumes 32 bit integer).
#define BITS32(O,S) ( MASK32(S) << (O) )

// extract S bits from (u_int32_t)W with offset O and shifts them O places to the right
#define EXTRACT32(W,O,S) ( ((W)>>(O)) & MASK32(S) )

// insert S bits with offset O from field F into word W (u_int32_t)
#define INSERT32(W,F,O,S) (/*(W)=*/ ( ((W) & (~BITS32(O,S)) ) | (((F) & MASK32(S))<<(O)) ))

#ifndef INSERTF
	#define INSERTF(W,O1,F,O2,S) (INSERT32(W, EXTRACT32(F, O2, S), O1, S) )
#endif


// according to Table 187 in the IB spec 1.2.1
typedef enum {
	SUBN_ADM_METHOD_SET    = 0x2,
	SUBN_ADM_METHOD_DELETE = 0x15
} subn_adm_method;

// Utilities for Umad Usage.
typedef enum {
	SUBN_ADM_COMPMASK_MGID         = (1ULL << 0),
	SUBN_ADM_COMPMASK_PORT_GID     = (1ULL << 1),
	SUBN_ADM_COMPMASK_Q_KEY	       = (1ULL << 2),
	SUBN_ADM_COMPMASK_P_KEY	       = (1ULL << 7),
	SUBN_ADM_COMPMASK_TCLASS       = (1ULL << 6),
	SUBN_ADM_COMPMASK_SL           = (1ULL << 12),
	SUBN_ADM_COMPMASK_FLOW_LABEL   = (1ULL << 13),
	SUBN_ADM_COMPMASK_JOIN_STATE   = (1ULL << 16),
} subn_adm_component_mask;

typedef enum {
	MCAST_IS_JOINED   = 1,
	MCAST_IS_ATTACHED = (1 << 1)
} mcast_state;


/************************************************************************
 *   Multicast data structures.						    			    *
 ************************************************************************/

// Needed parameters for creating a multiple multicast group entity.
struct mcast_parameters {
    int             	  num_qps_on_group;
	int					  is_user_mgid;
	int					  mcast_state;
	int 				  ib_port;
	uint16_t			  mlid;
	uint16_t			  base_mlid;
	const char			  *user_mgid;
	char				  *ib_devname;
	uint16_t 			  pkey;
	uint16_t			  sm_lid;
	uint8_t 			  sm_sl;
	union ibv_gid 		  port_gid;
	union ibv_gid 		  mgid;
	// In case it's a latency test.
	union ibv_gid         base_mgid;
	int is_2nd_mgid_used;
};

// according to Table 195 in the IB spec 1.2.1

struct sa_mad_packet_t {
	u_int8_t		mad_header_buf[24];
	u_int8_t		rmpp_header_buf[12];
	u_int64_t		SM_Key;
	u_int16_t		AttributeOffset;
	u_int16_t		Reserved1;
	u_int64_t		ComponentMask;
	u_int8_t		SubnetAdminData[200];
}__attribute__((packed));

/************************************************************************
 *   Multicast resources methods.					    			    *
 ************************************************************************/

/* set_multicast_gid .
 *
 * Description :
 *
 *  Sets the Multicast GID , and stores it in the "mgid" value of
 *  mcast resourcs. If the user requested for a specific MGID, which
 *  is stored in params->user_mgid (in this case params->is_user_mgid should be 1)
 *  than it will be his MGID, if not the library choose a default one.
 *
 * Parameters :
 *
 *  params            - The parameters of the machine
 *  my_dest ,rem_dest - The 2 sides that ends the connection.
 *
 * Return Value : 0 upon success. -1 if it fails.
 */
void set_multicast_gid(struct mcast_parameters *params,uint32_t qp_num,int is_client);


/* ctx_close_connection .
 *
 * Description :
 *
 *  Close the connection between the 2 machines.
 *  It performs an handshake to ensure the 2 sides are there.
 *
 * Parameters :
 *
 *  params            - The parameters of the machine
 *  my_dest ,rem_dest - The 2 sides that ends the connection.
 *
 * Return Value : 0 upon success. -1 if it fails.
 */
int join_multicast_group(subn_adm_method method,struct mcast_parameters *params);

#endif /* RDMA */

#endif /* MULTICAST_RESOURCES_H */
