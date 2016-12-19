#pragma once

/* Public External Interface to MSL
 * Author: Dhiraj Kalamkar
 * Date: 01-07-2016
 * vi: ts=4 sw=4 noet cino+=g0
 */

#include <cstddef>
#include <string>

namespace MSL
{
	/* Data Types supported by MSL */
	enum DataType {
		DT_FLOAT   = 1,
		DT_DOUBLE  = 2
	};

	/* Compute Operation Types
	 * Each Operation type defines specific relationship between input and output
	 * activations and associated weights
	 */
	enum OpType {
		COMP_OP_TYPE_CC     = 0, // Cross-Correlation - IFM and OFM independent and has weights
		COMP_OP_TYPE_BIAS   = 1, // BIAS - Same IFM and OFM (dependent) but has weights
		COMP_OP_TYPE_ACT    = 2, // Activation op - Same IFM and OFM and no weights
		COMP_OP_TYPE_POOL   = 3, // Activation op - Same IFM and OFM and no weights
		COMP_OP_TYPE_SPLIT  = 4, // OFM depends on IFM (=OFM1+OFM2...) and no weights
		COMP_OP_TYPE_CONCAT = 5, // OFM depends on IFM1+IFM2+... and no weights
		COMP_OP_TYPE_BCAST  = 6, // OFM1=IFM, OFM2=IFm, ... and no weights
		COMP_OP_TYPE_REDUCE = 7, // OFM=IFM1+IFM2+... and no weights
		COMP_OP_TYPE_DATA   = 8, // only OFM, no IFM
		COMP_OP_TYPE_EVAL   = 9  // only IFM, no OFM
	};

	class DistributionImpl;
	class ComputeOpRegInfoImpl;
	class ComputeOpImpl;
	class FeatureMapImpl;
	class WeightsImpl;
	class BlockInfoImpl;

	/* Initialize library, must be first the first call in library */
	int Init(int *argc, char **argv[]);

	/* Clean up and free all the internally allocated memory */
	int Finalize();

	/* Sets the Global Minibatch Size
	 * Should be called before any Compute Op creation
	 */
	int SetMinibatchSize(int globalMinibatchSize);

	/* Returns Global Node Id */
	int GetNodeId();

	/* Returns Node Number of self */
	int GetNumNodes();

	/* Global Barrier across all the nodes */
	void Barrier();

	/* Returns last error number */
	int GetError();

	/* MSL specific malloc and free to allocate communication buffers */
	void *Alloc(size_t sz, int align);
	void Free(void *);

	/* class to hold Block information for pack/unpack */
	class BlockInfo
	{
		BlockInfoImpl *p;
	public:
		BlockInfo(BlockInfoImpl *p_) : p(p_) { }
		~BlockInfo() { p = 0; }
		int MBStart();      // Start of Minibarch portion
		int MBLen();        // Length of Minibarch portion
		int FMStart();      // Start of Feature map portion
		int FMLen();        // Length of Feature map portion
		int FMSize();       // Size of each Feature map
		DataType FMType();  // Type of each Feature map element
		int BufOffset();    // Offset within comms buffer where to pack to/unpack from
	};

	class CommsBuf
	{
	private:
		void *p;
		size_t sz;
		bool isOwned;
	public:
		CommsBuf(size_t sz_) : sz(sz_), isOwned(false), p(0) { }

		~CommsBuf() {
			Free();
		}

		size_t Size() { return sz; }

		int Alloc() {
			if(sz == 0) return 0;
			p = MSL::Alloc(sz, 64);
			isOwned = true;
			if(p != NULL) return sz;
			return 0;
		}

		void Free() {
			if(isOwned && p != 0) {
				MSL::Free(p);
				p = 0;
			}
		}

		int SetPtr(void *ptr) {
			if(p != 0 && isOwned) {
				MSL::Free(p);
				isOwned = false;
			}
			p = ptr;
			return 0;
		}

		void *GetPtr() { return p; }
	};

	class FeatureMap
	{
	protected:
		FeatureMapImpl *p;
	public:
		FeatureMap(FeatureMapImpl *p_) : p(p_) { }
		~FeatureMap() { p = 0; }
		int GlobalLen();
		int GlobalOffset();
		int LocalLen();
		CommsBuf *CBuf();
		int NumPackBlocks();
		int NumUnpackBlocks();
		BlockInfo *GetPackBlock(int blockNumber);
		BlockInfo *GetUnpackBlock(int blockNumber);
		DataType GetDType();  // Type of feature map
		int FMSize();       // Size of each Feature map
		int DoComms(void *buf);
		int CommsStart(void *buf);
		void *CommsWait();
	};

	class Weights
	{
	protected:
		WeightsImpl *p;
	public:
		Weights(WeightsImpl *p_) : p(p_) { }
		~Weights() { p = 0; }
		int GlobalLen();
		int GlobalOffset();
		int LocalLen();
		int OwnedLen();
		int OwnedStart();
		int BiasStart(); // Offset at which Bias starts within LocalLen() i.e. Local Weights without Bias
		int BiasLen(); // Local Length of Bias
		int OwnedBiasStart(); // Offset at which Bias starts within OwnedLen() for SGD
		CommsBuf *CBuf();
		DataType GetDType();  // Type of weight
		int WTSize();       // Size of each Weight
		int DoDelWt(void *buf);
		int DoWtInc(void *buf);
		int CommsStartDelWt(void *buf);
		int CommsStartWtInc(void *buf);
		void *CommsWaitDelWt();
		void *CommsWaitWtInc();
	};

	/* Class to hold type of parallelism and parameter for hybrid parallelism
	 * nMBParts: Number of partitions on Minibatch
	 * nFMParts: Number of partitions on Input Feature Maps
	 * replicate: Whether to replicate data and compute if
	 *     nMBParts*nFMParts < numGlobalNodes and
	 *     numGlobalNodes % (nMBParts*nFMParts) == 0
	 */
	class Distribution
	{
	private:
		DistributionImpl *p;

	public:
		Distribution(int nMBParts, int nFMParts, bool replicate=true);
		~Distribution(void);
		int GetMBGroupSize();
		int GetFMGroupSize();
		int GetMBGroupId();
		int GetFMGroupId();
		DistributionImpl *GetImpl() { return p; }
	};

	/* Class to hold Compute operation's registartion information
	 * All the Inputs and Output feature maps and weights (if any) should be added
	 * and validated before calling constructor of ComputeOp
	 */
	class ComputeOpRegInfo
	{
	private:
		ComputeOpRegInfoImpl *p;
	public:
		ComputeOpRegInfo(OpType operationType, std::string name = "");
		~ComputeOpRegInfo();
		void SetName(const char *name); // For debugging purpose
		int AddInputFeatureMap(int numFeatureMaps, int featureMapSize, DataType featureMapType);
		int AddOutputFeatureMap(int numFeatureMaps, int featureMapSize, DataType featureMapType, bool fusedBias = false);
		int AddWeights(int numWeights, int weightSize, DataType weightType, bool distributedWeightUpdate = true);
		int Validate(Distribution *dist = NULL);
		ComputeOpRegInfoImpl *GetImpl() { return p; }
	};

	/*
	 * class to hold compute operation parameters and to hold relevant communication information
	 * FIXME: Add timing and stats related APIs
	 */
	class ComputeOp
	{
	private:
		ComputeOpImpl *p;

	public:
		ComputeOp(ComputeOpRegInfo *info, Distribution *dist);
		~ComputeOp();

		/* Sets the previous and next compute op in the graph
		 * id: index of input/output if op has multple inputs/outputs
		 * prevOpId/nextOpId: index of input/output within prev/next op
		 */
		int SetPrev(ComputeOp *prev, int id = 0, int prevOpId = -1);
		int SetNext(ComputeOp *next, int id = 0, int nextOpId = -1);

		/* Finalize need to be called bafore any comms operation can per performaned
		 * Finalize must be called after all SetPrev and SetNext calls are made
		 */
		int Finalize();

		/* Returns Compute op's distribution related inforation */
		std::string Name();
		Distribution *GetDistribution();// Ditsribution used by current ComputeOp
		int GlobalMinibatchLen();       // Length of global minibatch
		int LocalMinibatchLen();        // Length of local minibatch portion
		int GlobalMinibatchOffset();    // Start of local minibatch portion within global minibatch
		int NumInputFeatureMaps();
		FeatureMap *InputFeatureMap(int id = 0);
		int NumOutputFeatureMaps();
		FeatureMap *OutputFeatureMap(int id = 0);
		bool HasWeights();
		int NumWeights();
		MSL::Weights *Weights(int id = 0);
		/* AllocCommsBufs() allocates needed memory for communication */
		int AllocCommsBufs();                         // Internally allocate needed comms buffer
		void FreeCommsBufs();                         // Free internally allocated comms buffers

		ComputeOpImpl *GetImpl() { return p; }
	};

	void print_msl_time(void);
};

