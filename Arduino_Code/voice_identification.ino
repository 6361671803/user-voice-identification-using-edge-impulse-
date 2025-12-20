/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 * Licensed under the Apache License, Version 2.0
 */

#define EIDSP_QUANTIZE_FILTERBANK 0

#include <PDM.h>
#include <voice_identification_inferencing.h>

/** Audio buffers */
typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = false;

#define CONFIDENCE_THRESHOLD 0.60   // 60%

/* -------------------- SETUP -------------------- */
void setup() {
    Serial.begin(115200);
    while (!Serial);

    Serial.println("Edge Impulse Voice Identification");

    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tClasses: %d\n",
        sizeof(ei_classifier_inferencing_categories) /
        sizeof(ei_classifier_inferencing_categories[0]));

    if (!microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT)) {
        ei_printf("ERROR: Microphone init failed\n");
        while (1);
    }
}

/* -------------------- LOOP -------------------- */
void loop() {
    ei_printf("\nListening...\n");
    delay(2000);

    if (!microphone_inference_record()) {
        ei_printf("ERROR: Audio capture failed\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = microphone_audio_signal_get_data;

    ei_impulse_result_t result = {0};

    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERROR: Classifier failed (%d)\n", err);
        return;
    }

    /* -------- FIND MAX PROBABILITY -------- */
    float max_value = 0.0f;
    int max_index = -1;

    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > max_value) {
            max_value = result.classification[i].value;
            max_index = i;
        }
    }

    /* -------- FINAL CLEAN OUTPUT -------- */
    ei_printf("\n===== FINAL RESULT =====\n");

    if (max_value >= CONFIDENCE_THRESHOLD) {
        ei_printf("Detected Speaker: %s\n",
                  ei_classifier_inferencing_categories[max_index]);
        ei_printf("Confidence: %.2f %%\n", max_value * 100.0f);
    } else {
        ei_printf("Detected Speaker: UNKNOWN / NOISE\n");
        ei_printf("Confidence too low: %.2f %%\n", max_value * 100.0f);
    }

    ei_printf("========================\n");
}

/* -------------------- AUDIO CALLBACK -------------------- */
static void pdm_data_ready_inference_callback() {
    int bytesAvailable = PDM.available();
    int bytesRead = PDM.read((char *)sampleBuffer, bytesAvailable);

    if (!inference.buf_ready) {
        for (int i = 0; i < bytesRead >> 1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
                break;
            }
        }
    }
}

/* -------------------- MICROPHONE INIT -------------------- */
static bool microphone_inference_start(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (!inference.buffer) return false;

    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    PDM.onReceive(pdm_data_ready_inference_callback);
    PDM.setBufferSize(4096);

    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) return false;

    PDM.setGain(127);
    return true;
}

/* -------------------- RECORD AUDIO -------------------- */
static bool microphone_inference_record() {
    inference.buf_ready = 0;
    inference.buf_count = 0;

    while (!inference.buf_ready) {
        delay(10);
    }
    return true;
}

/* -------------------- SIGNAL DATA -------------------- */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
    return 0;
}

/* -------------------- SENSOR CHECK -------------------- */
#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid sensor for this model"
#endif
