#pragma once
#include <windows.h>
struct FrameMetadata {
    UINT64 swap, queue, device;
    UINT64 width;
    UINT height, format, buffer_count, current_index;
    UINT queue_type, valid_buffer;
};
