//
// Created by ben on 3/16/2026.
//
#include <random>
#include <chrono>
#include <algorithm>
#include <iterator>

//shuffles an array prior to sorting
template<typename T>
void shuffleArray(T* array, int n)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::shuffle(array, array + n, gen);
}


//for printing an array to a stream
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

