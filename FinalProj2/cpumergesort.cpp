#include <algorithm>
#include <vector>

namespace {

void mergeRanges(float* data, int left, int mid, int right, std::vector<float>& buffer) {
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

} // end anonymous namespace

// make sure this matches the float* signature perfectly
void sequentialMergeSort(float* data, int size) {
    if (data == nullptr || size < 2) {
        return;
    }

    std::vector<float> buffer(size);

    for (int width = 1; width < size; width *= 2) {
        for (int left = 0; left < size - width; left += 2 * width) {
            const int mid = left + width - 1;
            const int right = std::min(left + 2 * width - 1, size - 1);
            mergeRanges(data, left, mid, right, buffer);
        }
    }
}