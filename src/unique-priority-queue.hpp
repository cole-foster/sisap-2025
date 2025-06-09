#pragma once

#include <cmath>
#include <queue>
#include <unordered_set>
#include <utility>
typedef unsigned uint;


class UniquePriorityQueue {
public:
    std::priority_queue<std::pair<float,uint>> pq;         // Priority queue to store elements
    std::unordered_set<uint> seen;        // Set to track unique elements

    UniquePriorityQueue() {};


    UniquePriorityQueue(std::priority_queue<std::pair<float,uint>> existing_pq) {
        pq = existing_pq; // storing a copy
        
        // now add all to the seen set
        while (!existing_pq.empty()) {
            seen.insert(existing_pq.top().second);
            existing_pq.pop();
        }
    }

    // Insert element if not a duplicate
    bool push(std::pair<float,uint> pair) {
        // Check if the element is already in the set
        if (seen.find(pair.second) == seen.end()) {
            pq.emplace(pair);
            seen.insert(pair.second);
            return true; // Successfully added
        }
        return false; // Duplicate found, not added
    }

    bool emplace(float fval, uint uval) {
        // Check if the element is already in the set
        if (seen.find(uval) == seen.end()) {
            pq.emplace(fval,uval);
            seen.insert(uval);
            return true; // Successfully added
        }
        return false; // Duplicate found, not added
    }
    bool emplace(std::pair<float,uint> val) {
        return emplace(val.first, val.second);
    }

    // Access the top element
    const std::pair<float,uint>& top() const {
        return pq.top();
    }

    // Remove the top element
    void pop() {
        // Remove from the set and priority queue
        seen.erase(pq.top().second);
        pq.pop();
    }
    void pop_fast() {
        pq.pop();
    }

    // Check if the queue is empty
    bool empty() const {
        return pq.empty();
    }

    // Get the size of the queue
    std::size_t size() const {
        return pq.size();
    }
};
