#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <queue>

class ThreadPool{
 private:
    std::queue< boost::function< void() > > tasks_;
    boost::thread_group threads_;
    boost::mutex mutex_;
    boost::condition_variable condition_;
    boost::condition_variable completed_;
    bool running_;
    bool complete_;
    std::size_t available_;
    std::size_t total_;

 public:
    /// @brief Constructor.
    explicit ThreadPool(std::size_t pool_size)
        :  running_(true), complete_(true),
           available_(pool_size), total_(pool_size) {
        for ( std::size_t i = 0; i < pool_size; ++i ) {
            threads_.create_thread(
                boost::bind(&ThreadPool::main_loop, this));
        }
    }

    /// @brief Destructor.
    ~ThreadPool() {
        // Set running flag to false then notify all threads.
        {
            boost::unique_lock< boost::mutex > lock(mutex_);
            running_ = false;
            condition_.notify_all();
        }

        try {
            threads_.join_all();
        }
        // Suppress all exceptions.
        catch (const std::exception&) {}
    }

    /// @brief Add task to the thread pool if a thread is currently available.
    template <typename Task>
    void runTask(Task task) {
        boost::unique_lock<boost::mutex> lock(mutex_);

        // Set task and signal condition variable so that a worker thread will
        // wake up and use the task.
        tasks_.push(boost::function<void()>(task));
        complete_ = false;
        condition_.notify_one();
    }

    /// @brief Wait for queue to be empty
    void waitWorkComplete() {
        boost::unique_lock<boost::mutex> lock(mutex_);
        if (!complete_)
            completed_.wait(lock);
    }

 private:
    /// @brief Entry point for pool threads.
    void main_loop() {
        while (running_) {
            // Wait on condition variable while the task is empty and
            // the pool is still running.
            boost::unique_lock<boost::mutex> lock(mutex_);
            while (tasks_.empty() && running_) {
                condition_.wait(lock);
            }
            // If pool is no longer running, break out of loop.
            if (!running_) break;

            // Copy task locally and remove from the queue.  This is
            // done within its own scope so that the task object is
            // destructed immediately after running the task.  This is
            // useful in the event that the function contains
            // shared_ptr arguments bound via bind.
            {
                boost::function< void() > task = tasks_.front();
                tasks_.pop();
                // Decrement count, indicating thread is no longer available.
                --available_;

                lock.unlock();

                // Run the task.
                try {
                    task();
                }
                // Suppress all exceptions.
                catch ( const std::exception& ) {}

                // Update status of empty, maybe
                // Need to recover the lock first
                lock.lock();

                // Increment count, indicating thread is available.
                ++available_;
                if (tasks_.empty() && available_ == total_) {
                    complete_ = true;
                    completed_.notify_one();
                }
            }
        }  // while running_
    }
};

#endif
