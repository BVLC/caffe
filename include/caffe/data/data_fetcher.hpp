// Copyright 2014 BVLC and contributors.
/*
 * Adapted from the DataFetcher of cxxnet
 */
#ifndef CAFFE_UTIL_DATA_FETCHER_H_
#define CAFFE_UTIL_DATA_FETCHER_H_

#include <string>
#include <vector>

#include <boost/thread.hpp>

#include "caffe/proto/caffe.pb.h"

#include "caffe/blob.hpp"

namespace caffe {
using std::string;
using std::vector;
/*!
 * \brief buffered loading iterator that uses multithread
 * this template method will assume the following paramters
 * \tparam Elem elememt type to be buffered
 * \tparam ElemFactory factory_ type to implement in order to use thread buffer
 */
template<typename Elem, typename ElemFactory>
class DataFetcher {
public:
	/*!\brief constructor */
	DataFetcher() {
		this->init_end_ = false;
		this->buf_size = 30;
	}
	~DataFetcher(void) {
		if (init_end_)
			this->Destroy();
	}
	/*!\brief set parameter, will also pass the parameter to factory_ */
	inline void SetParam(const char *name, const char *val) {
		if (!strcmp(name, "buffer_size"))
			buf_size = atoi(val);
		factory_.SetParam(name, val);
	}
	/*!
	 * \brief initalize the buffered iterator
	 * \param param a initialize parameter that will pass to factory_, ignore it if not necessary
	 * \return false if the initlization can't be done, e.g. buffer file hasn't been created
	 */
	inline bool Init(void) {
		if (!factory_.Init())
			return false;
		for (int i = 0; i < buf_size; i++) {
			bufA_.push_back(factory_.Create());
			bufB_.push_back(factory_.Create());
		}
		this->init_end_ = true;
		this->StartLoader();
		return true;
	}

	/*!\brief place the iterator before first value */
	inline void BeforeFirst(void) {
		// wait till last loader end
		loading_end_.Wait();
		// critcal zone
		current_buf_ = 1;
		factory_.BeforeFirst();
		// reset terminate limit
		endA_ = endB_ = buf_size;
		// wake up loader for first part
		loading_need_.Post();
		// wait til first part is loaded
		loading_end_.Wait();
		// set current buf to right value
		current_buf_ = 0;
		// wake loader for next part
		data_loaded_ = false;
		loading_need_.Post();
		// set buffer value
		buf_index_ = 0;
	}

	/*! \brief destroy the buffer iterator, will deallocate the buffer */
	inline void Destroy(void) {
		// wait until the signal is consumed
		this->destroy_signal_ = true;
		loading_need_.Post();
		loader_thread_.Join();
		loading_need_.Destroy();
		loading_end_.Destroy();

		for (size_t i = 0; i < bufA_.size(); i++) {
			factory_.FreeSpace(bufA_[i]);
		}
		for (size_t i = 0; i < bufB_.size(); i++) {
			factory_.FreeSpace(bufB_[i]);
		}
		bufA_.clear();
		bufB_.clear();
		factory_.Destroy();
		this->init_end_ = false;
	}

	/*!
	 * \brief get the next element needed in buffer
	 * \param elem element to store into
	 * \return whether reaches end of data
	 */
	inline bool HasNext(Elem &elem) {
		// end of buffer try to switch
		if (buf_index_ == buf_size) {
			this->SwitchBuffer();
			buf_index_ = 0;
		}
		if (buf_index_ >= (current_buf_ ? endA_ : endB_)) {
			return false;
		}
		std::vector<Elem> &buf = current_buf_ ? bufA_ : bufB_;
		elem = buf[buf_index_];
		buf_index_++;
		return true;
	}
	/*!
	 * \brief get the factory_ object
	 */
	inline ElemFactory &get_factory() {
		return factory_;
	}
private:
	/*!
	 * \brief slave thread
	 * this implementation is like producer-consumer style
	 */
	inline void RunLoader() {
		while (!destroy_signal_) {
			// sleep until loading is needed
			loading_need_.Wait();

			std::vector<Elem> &buf = current_buf_ ? bufB_ : bufA_;
			int i;
			for (i = 0; i < buf_size; i++) {
				if (!factory_.LoadNext(buf[i])) {
					int &end = current_buf_ ? endB_ : endA_;
					end = i; // marks the termination
					break;
				}
			}

			// signal that loading is done
			data_loaded_ = true;
			loading_end_.Post();
		}
	}
	/*!\brief entry point of loader thread */
	inline static CXXNET_THREAD_PREFIX LoaderEntry(void *pthread) {
		static_cast<DataFetcher<Elem, ElemFactory>*>(pthread)->RunLoader();
		ThreadExit (NULL);
		return NULL;
	}
	/*!\brief start loader thread */
	inline void StartLoader() {
		destroy_signal_ = false;
		// set param
		current_buf_ = 1;

		loading_need_.Init(1);
		loading_end_.Init(0);
		// reset terminate limit
		endA_ = endB_ = buf_size;
		loader_thread_.Start(LoaderEntry, this);
		// wait until first part of data is loaded
		loading_end_.Wait();
		// set current buf to right value
		current_buf_ = 0;
		// wake loader for next part
		data_loaded_ = false;
		loading_need_.Post();

		buf_index_ = 0;
	}
	/*!\brief switch double buffer */
	inline void SwitchBuffer() {
		loading_end_.Wait();
		// loader shall be sleep now, critcal zone!
		current_buf_ = !current_buf_;
		// wake up loader
		data_loaded_ = false;
		loading_need_.Post();
	}
public:
	// size of buffer
	int buf_size;
private:
	// factory_ object used to load configures
	ElemFactory factory_;
	// index in current buffer
	int buf_index_;
	// indicate which one is current buffer
	int current_buf_;
	// max limit of visit, also marks termination
	int endA_, endB_;
	// double buffer, one is accessed by loader
	// the other is accessed by consumer
	// buffer of the data
	std::vector<Elem> bufA_, bufB_;
	// initialization end
	bool init_end_;
	// singal whether the data is loaded
	bool data_loaded_;
	// signal to kill the thread
	bool destroy_signal_;
	// thread object
	Thread loader_thread_;
	// signal of the buffer
	Semaphore loading_end_, loading_need_;
};

}  // namespace caffe

#endif   // CAFFE_UTIL_DATA_FETCHER_H_
