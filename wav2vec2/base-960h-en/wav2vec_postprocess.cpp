/*
 * wav2vec2 postprocess implementation (English model)
 * Based on facebook/wav2vec2-base-960h
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <android/log.h>
#include "wav2vec_postprocess.h"
#include "../common/Utils.h"

// wav2vec2 English model output size
#define VOCAB_SIZE 32
#define SEQUENCE_LENGTH 249  // Model output: [1, 249, 32] for 5 seconds input

// Special token IDs (from facebook/wav2vec2-base-960h)
#define PAD_TOKEN_ID 0
#define BOS_TOKEN_ID 1
#define EOS_TOKEN_ID 2
#define UNK_TOKEN_ID 3
#define WORD_DELIMITER_ID 4  // "|" (space)

// English token mapping (facebook/wav2vec2-base-960h tokenizer)
static const char* get_token_text(int token_id) {
    // Special tokens
    if (token_id == PAD_TOKEN_ID) return "";
    if (token_id == BOS_TOKEN_ID) return "";
    if (token_id == EOS_TOKEN_ID) return "";
    if (token_id == UNK_TOKEN_ID) return "";
    if (token_id == WORD_DELIMITER_ID) return " ";
    
    // English alphabet tokens (uppercase)
    switch (token_id) {
        case 5: return "E";
        case 6: return "T";
        case 7: return "A";
        case 8: return "O";
        case 9: return "N";
        case 10: return "I";
        case 11: return "H";
        case 12: return "S";
        case 13: return "R";
        case 14: return "D";
        case 15: return "L";
        case 16: return "U";
        case 17: return "M";
        case 18: return "W";
        case 19: return "C";
        case 20: return "F";
        case 21: return "G";
        case 22: return "Y";
        case 23: return "P";
        case 24: return "B";
        case 25: return "V";
        case 26: return "K";
        case 27: return "'";
        case 28: return "X";
        case 29: return "J";
        case 30: return "Q";
        case 31: return "Z";
        default: return "";
    }
}

// Remove consecutive duplicate tokens (CTC decoding)
void removeDuplicateTokens(int* tokens, int* length) {
    if (*length <= 1) {
        return;
    }
    
    int* result = (int*)malloc(*length * sizeof(int));
    int resultIndex = 0;
    result[resultIndex++] = tokens[0];
    
    for (int i = 1; i < *length; i++) {
        if (tokens[i] != tokens[i - 1]) {
            result[resultIndex++] = tokens[i];
        }
    }
    
    memcpy(tokens, result, resultIndex * sizeof(int));
    *length = resultIndex;
    free(result);
}

float wav2vec_postprocess(float *tensor_data, char *result_text) {
    // Model output: [1, 249, 32] in memory layout [seq, vocab]
    // tensor_data[seq * VOCAB_SIZE + vocab]
    int sequence_length = SEQUENCE_LENGTH;

    LOGD("=== wav2vec2 Postprocess Start ===");
    LOGD("Sequence length: %d, Vocab size: %d", sequence_length, VOCAB_SIZE);
    LOGD("Tensor data pointer: %p", tensor_data);

    // Greedy decoding: find max probability token at each time step
    // Memory layout is [seq, vocab]: tensor_data[seq * VOCAB_SIZE + vocab]
    int predicted_tokens[SEQUENCE_LENGTH];
    float total_confidence = 0.0f;

    LOGD("=== CTC Decoding ===");

    for (int seq = 0; seq < sequence_length; seq++) {
        int max_index = 0;
        float max_logit = tensor_data[seq * VOCAB_SIZE + 0];

        // Find max logit across all vocab for this sequence position
        for (int vocab = 1; vocab < VOCAB_SIZE; vocab++) {
            float logit = tensor_data[seq * VOCAB_SIZE + vocab];
            if (logit > max_logit) {
                max_logit = logit;
                max_index = vocab;
            }
        }

        // Calculate softmax sum for numerical stability
        float sum_exp = 0.0f;
        for (int vocab = 0; vocab < VOCAB_SIZE; vocab++) {
            float logit = tensor_data[seq * VOCAB_SIZE + vocab];
            sum_exp += expf(logit - max_logit);
        }
        
        // Calculate probability of max token
        float max_prob = 1.0f / sum_exp;
        
        predicted_tokens[seq] = max_index;
        total_confidence += max_prob;
        
        // Debug output for first 10 steps
        if (seq < 10) {
            LOGD("Step %d: token=%d ('%s'), logit=%.3f, prob=%.3f", 
                 seq, max_index, get_token_text(max_index), max_logit, max_prob);
        }
    }
    
    LOGD("=== Predicted Tokens (first 30) ===");
    for (int i = 0; i < 30 && i < sequence_length; i++) {
        const char* token_str = get_token_text(predicted_tokens[i]);
        if (strlen(token_str) > 0) {
            LOGD("Token[%d] = %d ('%s')", i, predicted_tokens[i], token_str);
        } else {
            LOGD("Token[%d] = %d (PAD/special)", i, predicted_tokens[i]);
        }
    }
    
    // Count PAD tokens
    int pad_count = 0;
    for (int i = 0; i < sequence_length; i++) {
        if (predicted_tokens[i] == PAD_TOKEN_ID) pad_count++;
    }
    LOGD("PAD token count: %d/%d", pad_count, sequence_length);
    
    // Calculate average confidence
    float confidence = total_confidence / sequence_length;
    LOGD("Average confidence: %.3f", confidence);
    
    // CTC decoding: remove consecutive duplicate tokens
    LOGD("Before duplicate removal: %d tokens", sequence_length);
    removeDuplicateTokens(predicted_tokens, &sequence_length);
    LOGD("After duplicate removal: %d tokens", sequence_length);
    
    // Convert tokens to text
    result_text[0] = '\0';
    char temp_buffer[2048] = {0};
    
    LOGD("=== Token to Text Conversion ===");
    int valid_tokens = 0;
    
    for (int i = 0; i < sequence_length; i++) {
        const char* token_text = get_token_text(predicted_tokens[i]);
        if (strlen(token_text) > 0) {
            strcat(temp_buffer, token_text);
            valid_tokens++;
            if (valid_tokens <= 20) {
                LOGD("Token %d ('%s') -> '%s'", predicted_tokens[i], get_token_text(predicted_tokens[i]), token_text);
            }
        }
    }
    
    LOGD("Valid tokens: %d, Text length: %d", valid_tokens, (int)strlen(temp_buffer));
    
    // Copy result (prevent buffer overflow)
    strncpy(result_text, temp_buffer, 1023);
    result_text[1023] = '\0';
    
    // If empty result, set placeholder
    if (strlen(result_text) == 0) {
        strcpy(result_text, "[NO SPEECH DETECTED]");
        LOGD("Empty result, using placeholder");
    }
    
    LOGD("=== wav2vec2 Postprocess Complete ===");
    LOGD("Final result: '%s' (confidence: %.3f)", result_text, confidence);
    
    return confidence;
}