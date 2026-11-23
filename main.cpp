// main.cpp — RKLLM 1.2.x (LLMHandle / RKLLMCallback struct / RKLLMInferParam)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include "rkllm.h"

static std::mutex mtx;
static std::condition_variable cv;
static bool finished = false;

// LLMResultCallback
static int result_cb(RKLLMResult* result, void* userdata, LLMCallState state) {
    switch (state) {
    case RKLLM_RUN_NORMAL:
        if (result && result->text) { fputs(result->text, stdout); fflush(stdout); }
        break;
    case RKLLM_RUN_ERROR:
        fprintf(stderr, "\n[rkllm error]\n");
        { std::lock_guard<std::mutex> lk(mtx); finished = true; }
        cv.notify_one();
        break;
    case RKLLM_RUN_FINISH:
        { std::lock_guard<std::mutex> lk(mtx); finished = true; }
        cv.notify_one();
        break;
    default:
        break;   // RKLLM_RUN_WAITING 등
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.rkllm [max_context_len]\n", argv[0]);
        return 1;
    }

    // ★ 이 버전의 기본값 생성 함수
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path         = argv[1];
    param.max_context_len    = (argc > 2) ? atoi(argv[2]) : 2048;
    param.max_new_tokens     = 1024;
    param.top_k              = 1;
    param.top_p              = 0.9f;
    param.temperature        = 0.7f;
    param.repeat_penalty     = 1.1f;
    param.skip_special_token = true;

    // ★ 콜백은 구조체. tokenizer/embed 콜백은 NULL로 둠
    //    (내장 토크나이저가 있는 일반 .rkllm 모델은 이대로 OK)
    RKLLMCallback cb{};
    cb.result_callback = result_cb;
    cb.result_userdata = nullptr;

    LLMHandle handle = nullptr;
    if (rkllm_init(&handle, &param, &cb) != 0) {
        fprintf(stderr, "rkllm_init 실패: 모델/SDK 버전, NPU 드라이버 확인\n");
        return -1;
    }

    // Qwen 계열 템플릿 (다른 모델이면 토크나이저 설정에 맞게 수정)
    rkllm_set_chat_template(handle,
        "You are a helpful assistant.",
        "<|im_start|>user\n",
        "<|im_end|>\n<|im_start|>assistant\n");

    // ★ 추론 파라미터
    RKLLMInferParam infer{};
    infer.mode           = RKLLM_INFER_GENERATE;
    infer.keep_history   = 1;   // 멀티턴 유지
    infer.max_new_tokens = 0;   // 0 이하 = init 값 사용

    bool fresh = false;
    std::string line;
    while (true) {
        printf("\nUser> "); fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line == "/exit" || line == "/quit") break;
        if (line == "/new") { fresh = true; printf("(대화 히스토리 초기화)\n"); continue; }
        if (line.empty()) continue;

        RKLLMInput input{};
        input.input_type   = RKLLM_INPUT_PROMPT;
        input.prompt_input = (char*)line.c_str();

        infer.keep_history = fresh ? 0 : 1;
        fresh = false;

        { std::lock_guard<std::mutex> lk(mtx); finished = false; }
        printf("Bot> "); fflush(stdout);

        int ret = rkllm_run(handle, &input, &infer, nullptr);
        if (ret != 0) { fprintf(stderr, "\nrkllm_run 실패: %d\n", ret); continue; }

        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, []{ return finished; });
        printf("\n");
    }

    rkllm_destroy(handle);
    return 0;
}