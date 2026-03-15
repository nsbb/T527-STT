#ifndef __WAV2VEC_POST_PROCESS_H__
#define __WAV2VEC_POST_PROCESS_H__

#ifdef __cplusplus
extern "C" {
#endif

/**
 * wav2vec postprocess function (English model)
 * @param tensor_data NPU output tensor data
 * @param result_text result text buffer
 * @return confidence score
 */
float wav2vec_postprocess(float *tensor_data, char *result_text);

#ifdef __cplusplus
}
#endif

#endif