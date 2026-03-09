#include <vector>


using namespace std;
namespace {

void mergeRanges(int* data, int left, int mid, int right, vector<int>& buffer) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (data[i] <= data[j]) {
            buffer[k++] = data[i++];
        } else {
            buffer[k++] = data[j++];
        }
    }

    while (i <= mid) {
        buffer[k++] = data[i++];
    }

    while (j <= right) {
        buffer[k++] = data[j++];
    }

    for (int index = left; index <= right; ++index) {
        data[index] = buffer[index];
    }
}

void mergeSortRecursive(int* data, int left, int right, vector<int>& buffer) {
    if (left >= right) {
        return;
    }

    const int mid = left + (right - left) / 2;
    mergeSortRecursive(data, left, mid, buffer);
    mergeSortRecursive(data, mid + 1, right, buffer);
    mergeRanges(data, left, mid, right, buffer);
}

}  

void sequentialMergeSort(int* data, int size) {
    if (data == nullptr || size < 2) {
        return;
    }

    vector<int> buffer(size);
    mergeSortRecursive(data, 0, size - 1, buffer);
}
