#include "ThreadPool.h"


ThreadPool::ThreadPool() {

    m_numThreads = 1;
    processing = 0;

}

ThreadPool::ThreadPool(int64_t numThreads) {

    m_numThreads = numThreads;
    processing = 0;

}

ThreadPool::~ThreadPool() {

}

void ThreadPool::ThreadLoop() {
    while (true)
    {
        std::unique_lock<std::mutex> latch(queue_mutex);
        mutex_condition.wait(latch, [this] { return !jobs.empty() || should_terminate; });

        if (!jobs.empty())
        {
            ++processing; // We got work, so add to a processing count. Ensures we don't have threads running after the queue is empty

            auto job = jobs.front();
            jobs.pop_front();

            latch.unlock(); // release lock, run function asynchronously

            job();

            latch.lock();
            --processing; // remove job from processing count
            job_finished.notify_one(); // ping the job finished, which will cause it to check if the queue is empty and no jobs are still processing in waitForJobsToFinish()
        }
        else if (should_terminate)
        {
            return;
        }
    }
}


void ThreadPool::start() {
    threads.resize(m_numThreads);
    for (uint64_t i = 0; i < m_numThreads; i++) {
        threads.at(i) = std::thread(&ThreadPool::ThreadLoop, this);
    }
    std::cout << "made " << threads.size() << " threads in pool" << std::endl;
}


void ThreadPool::queueJob(std::function<void()> job) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        jobs.emplace_back(job);
        mutex_condition.notify_one();
    }
}

void ThreadPool::waitForJobsToFinish() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        job_finished.wait(lock, [this]() { return jobs.empty() && (processing == 0); });
    }
}

bool ThreadPool::busy() {
    bool poolbusy;
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        poolbusy = !jobs.empty();
    }
    return poolbusy;
}

void ThreadPool::stop() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        should_terminate = true;
    }
    mutex_condition.notify_all();
    for (std::thread& active_thread : threads) {
        active_thread.join();
    }
    threads.clear();
}