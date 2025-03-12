#ifndef THREADPOOL_H
#define THREADPOOL_H


#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
public:

    // Default constructor
    ThreadPool() noexcept;

    // Overloaded constructor allowing specific number of threads
    explicit ThreadPool(int64_t numThreads) noexcept;

    // Destructor
    ~ThreadPool() noexcept;

    // Non-copyable & non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Start the thread pool
    void start() noexcept;

    // Queue a job for execution
    void queueJob(std::function<void()>&& job) noexcept;

    // Wait for all queued jobs to complete
    void waitForJobsToFinish() noexcept;

    // Stop the thread pool
    void stop() noexcept;

    // Check if the thread pool has any jobs currently queued
    [[nodiscard]] bool busy() const noexcept;

private:
    // The thread loop processing function
    void threadLoop() noexcept;

    // Number of threads in pool
    const int64_t m_numThreads;

    // Flag indicating if threads should terminate execution
    std::atomic<bool> m_shouldTerminate{false};

    // Mutex for job queue access
    mutable std::mutex m_queueMutex;

    // Condition variable for job availability
    std::condition_variable m_jobCondition;

    // Condition variable for job completion
    std::condition_variable m_jobFinishedCondition;

    // Thread pool
    std::vector<std::thread> m_threads;

    // Job queue
    std::deque<std::function<void()>> m_jobs;

    // Number of jobs currently being processed
    std::atomic<uint64_t> m_processing{0};

};

#endif