#include <dfmarker_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* This example matches the synthetic 640x480 SDK demo configuration. */
struct callback_state {
    unsigned count;
    int invalid;
};

static void on_result(const dfmarker_result_t *result, void *user_data) {
    struct callback_state *state = (struct callback_state *)user_data;
    if (!result || result->frame_index != state->count ||
        result->timestamp_ns != (uint64_t)state->count * UINT64_C(33333333) ||
        result->quality != DFMARKER_LOST || result->has_pose) {
        state->invalid = 1;
    }
    ++state->count;
}

int main(int argc, char **argv) {
    DFMARKER_HANDLE handle = NULL;
    dfmarker_status_t status = DFMARKER_ERROR;
    struct callback_state state = {0, 0};
    const size_t size = 640u * 480u;
    uint8_t *pixels;
    int success;
    dfmarker_image_t frame;
    if (argc != 2) {
        fprintf(stderr, "usage: c_api_smoke sdk_demo.json\n");
        return 2;
    }
    pixels = (uint8_t *)malloc(size);
    if (!pixels) return 1;
    memset(pixels, 255, size);
    memset(&frame, 0, sizeof(frame));
    frame.width = 640;
    frame.height = 480;
    frame.stride_bytes = 640;
    frame.data = pixels;
    frame.data_size_bytes = size;
    frame.format = DFMARKER_MONO8;
    success = dfmarker_create(argv[1], &handle) &&
        dfmarker_get_status(handle, &status) && status == DFMARKER_CREATED &&
        dfmarker_register_result_callback(handle, on_result, &state) &&
        dfmarker_start(handle) &&
        dfmarker_get_status(handle, &status) && status == DFMARKER_RUNNING;
    if (success) success = dfmarker_feed_image(handle, &frame);
    frame.timestamp_ns = UINT64_C(33333333);
    if (success) success = dfmarker_feed_image(handle, &frame);
    if (success) success = state.count == 2 && !state.invalid &&
        dfmarker_stop(handle) && dfmarker_get_status(handle, &status) &&
        status == DFMARKER_STOPPED;
    if (handle && !dfmarker_destroy(&handle)) success = 0;
    if (handle) success = 0;
    free(pixels);
    puts(success ? "PASS: installed C consumer, lifecycle and LOST callbacks" :
                   "FAIL: C SDK smoke");
    return success ? 0 : 1;
}
