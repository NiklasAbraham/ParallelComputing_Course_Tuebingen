#include "sample_kernel.h"
#include <stdio.h>

int main(void) {
    struct futhark_context_config *cfg = futhark_context_config_new();
    if (cfg == NULL) {
        fprintf(stderr, "Failed to create config\n");
        return 1;
    }
    futhark_context_config_add_nvrtc_option(cfg, "--gpu-architecture=sm_86");
    struct futhark_context *ctx = futhark_context_new(cfg);
    if (ctx == NULL) {
        fprintf(stderr, "Failed to create context\n");
        return 2;
    }
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}

