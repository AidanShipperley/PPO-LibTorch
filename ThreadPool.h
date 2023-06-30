#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <iostream>
#include <vector>
#include <deque>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <thread>

// https://stackoverflow.com/questions/15752659/thread-pooling-in-c11
class ThreadPool {
public:

    ThreadPool();
    ThreadPool(int numThreads);
    ~ThreadPool();
    void start();
    void queueJob(std::function<void()> job);
    void waitForJobsToFinish();
    void stop();
    bool busy();
    void ThreadLoop();

    int m_numThreads;

private:

    bool should_terminate = false;           // Tells threads to stop looking for jobs
    std::mutex queue_mutex;                  // Prevents data races to the job queue
    std::condition_variable mutex_condition; // Allows threads to wait on new jobs or termination
    std::condition_variable job_finished;
    std::vector<std::thread> threads;
    std::deque<std::function<void()>> jobs;
    unsigned int processing;

};

#endif