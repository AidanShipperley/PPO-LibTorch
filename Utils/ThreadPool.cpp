#include "ThreadPool.h"


ThreadPool::ThreadPool() noexcept
    : m_numThreads(1) { 
}

ThreadPool::ThreadPool(int64_t numThreads) noexcept
    : m_numThreads(numThreads) {
}

ThreadPool::~ThreadPool() noexcept {
    // Ensure all threads are stopped when object is destroyed
    if (!m_shouldTerminate && !m_threads.empty()) {
        stop();
    }
}

void ThreadPool::threadLoop() noexcept {
    while (true) {
        std::function<void()> job;

        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
        
            // Wait for job or for termination signal
            m_jobCondition.wait(lock, [this] { 
                return !m_jobs.empty() || m_shouldTerminate;
            });

            // Check for jobs first
            if (!m_jobs.empty()) {
                // Increment the processing counter
                ++m_processing; 

                // Get job from queue via move to avoid copying
                job = std::move(m_jobs.front());
                m_jobs.pop_front();

                // Release lock before executing job for better concurrency
                lock.unlock();

                try {
                    // Exec job
                    job();
                } catch (...) {
                    // Prevent exceptions from escaping and crashing the thread
                    std::cerr << "Exception occurred in thread pool job" << std::endl;
                }

                // Reacquire lock to update processing counter
                lock.lock();

                // Decrement processing counter
                --m_processing;

                // Notify job completion
                m_jobFinishedCondition.notify_all();
            }
            // If no jobs and should terminate, exit thread
            else if (m_shouldTerminate) {
                return;
            }
        }
    }
}


void ThreadPool::start() noexcept {
    // Reset termination flag
    m_shouldTerminate = false;

    // Reserve space for threads to avoid reallocation
    m_threads.reserve(static_cast<size_t>(m_numThreads));

    // Create and start threads in pool
    for (int64_t i = 0; i < m_numThreads; ++i) {
        m_threads.emplace_back(&ThreadPool::threadLoop, this);
    }

    std::cout << "Created " << m_threads.size() << " threads in pool" << std::endl;
}


void ThreadPool::queueJob(std::function<void()>&& job) noexcept {
    {
        // Use scoped_lock during queue
        std::scoped_lock lock(m_queueMutex);

        // Move job into queue for efficiency
        m_jobs.emplace_back(std::move(job));
    }

    // Notify one waiting thread of job availability
    m_jobCondition.notify_one();
}


void ThreadPool::waitForJobsToFinish() noexcept {
    std::unique_lock<std::mutex> lock(m_queueMutex);

    // Wait until all jobs are complete
    m_jobFinishedCondition.wait(lock, [this] {
        return m_jobs.empty() && (m_processing == 0);
    });    
}

bool ThreadPool::busy() const noexcept {

    // Use scoped lock while checking if pool is empty
    std::scoped_lock lock(m_queueMutex);

    // Return true if jobs are queued
    return !m_jobs.empty();

}

void ThreadPool::stop() noexcept {
    {
        // Use scoped lock when setting termination lock
        std::scoped_lock lock(m_queueMutex);
        
        // Set termination flag
        m_shouldTerminate = true;
    }

    // Notify all waiting threads to check termination condition
    m_jobCondition.notify_all();

    // Join all threads
    for (std::thread& thread : m_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Clear out the thread vector
    m_threads.clear();
}
