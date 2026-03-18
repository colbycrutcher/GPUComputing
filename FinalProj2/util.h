#ifndef FINALPROJ_UTIL_H
#define FINALPROJ_UTIL_H

#include <random>
#include <chrono>
#include <algorithm>
#include <iterator>
#include <iostream>

template<typename T>
void shuffleArray(T* array, int n)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(array, array + n, gen);
}

template<typename T>
void printArrayTo(std::ostream &os, T* array, int n)
{
    os << "[ ";
    for (int i = 0; i < n ; i++)
    {
        os << array[i] << " ";
    }
    os << "]\n";
}

#endif // FINALPROJ_UTIL_H