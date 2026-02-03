// uwpspatialsoundtest.cpp : This file contains the 'main' function. Program execution begins and ends there.
// Most (if not all) of these are from https://learn.microsoft.com/en-us/windows/win32/coreaudio/render-spatial-sound-using-spatial-audio-objects

#include <iostream>
#include <fstream>
#include <random>
#include <stdlib.h>
#include <locale.h>
#include <thread>
#include <stdio.h>
#include <string>
#include <filesystem>
#include <io.h>
#include <fcntl.h>
#include <algorithm>
#include <chrono>

#include <windows.h>
#include <windows.foundation.h>
#include <windowsnumerics.h>
#include <wrl/wrappers/corewrappers.h>
#include <wrl/client.h>
#include <SpatialAudioClient.h>
#include <mmdeviceapi.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <ppl.h>

#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

// Audio object struct
struct Speaker3dObject {
    Microsoft::WRL::ComPtr<ISpatialAudioObject> audioObject;
	Windows::Foundation::Numerics::float3 position; // In meters
    float volume; // 0.0 to 1.0
    std::vector<float>* wavSample;
    UINT totalFrameCount;
    UINT currFrameIndex;   // Since chunks of audio are written to buffer
};

enum SpeakerChannels {
    SpeakerChannel_LeftMRH,
    SpeakerChannel_RightMRH,
    SpeakerChannel_MonoDistortion,
    SpeakerChannel_MonoLFE,
};

// Writes frameCount samples from PCM sample vector to buffer
UINT WriteToAudioObjectBuffer(
    FLOAT* buffer,
    UINT frameCount,
    const std::vector<float> &data,
    UINT &pos
) {
    UINT writeLen = min(frameCount, static_cast<UINT>(max(0, data.size() - pos)));
    if (writeLen > 0) {
        memcpy(buffer, data.data() + pos, sizeof(float) * writeLen);
		pos += writeLen;
    }
    if (writeLen < frameCount) {
		memset(buffer + writeLen, 0, sizeof(float) * (frameCount - writeLen)); // Zero-fill remainder
    }
    return writeLen; // Required for ISpatialAudioObjectBase::SetEndOfStream
}

inline void BiquadFilter(
    std::vector<float>& wavSamples,
    float a1, float a2, float b0, float b1, float b2
) {
    float y = 0.0f;
    float prevTerm = 0.0f;
    float prevPrevTerm = 0.0f;
    for (size_t i = 0; i < wavSamples.size(); i++) {
        y = b0 * wavSamples[i] + prevTerm;
        prevTerm = b1 * wavSamples[i] - a1 * y + prevPrevTerm;
        prevPrevTerm = b2 * wavSamples[i] - a2 * y;
        wavSamples[i] = y;
    }
}

// Soft saturator for "boominess"
inline void SoftSaturator(std::vector<float>& wavSamples, float drive) {
    concurrency::parallel_for(size_t(0), wavSamples.size(), [&](size_t i) {
        wavSamples[i] = std::tanh(drive * wavSamples[i]);
    });
}

// Low-pass filter, uses two second-order IIR filters
std::vector<float> LowPassFilter(
    const std::vector<float>& wavSamples,
    float freqCutoff,
    UINT sampleRate,
    float q=0.70710678
) {
    std::vector<float> outWavSamples = wavSamples;

    // Approximation of w0 from https://dsp.stackexchange.com/questions/28308/3-db-cut-off-frequency-of-exponentially-weighted-moving-average-filter/28314#28314
    // Biquad filter from https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
    // The Q values are from 4th-order Butterworth low-pass Perplexity got that I don't feel qualified to understand
    float w0 = 2.0f * 3.1415926f * freqCutoff / sampleRate;
    float cosw0 = std::cos(w0);
    float sinw0 = std::sin(w0);
    float alpha0 = sinw0 * q / 2;

    float a0 = 1.0f + alpha0;
    float a1 = -2.0f * cosw0;
    float a2 = 1.0f - alpha0;
    float b1 = 1.0f - cosw0;
    float b02 = b1 / 2;
    a1 /= a0, a2 /= a0, b1 /= a0, b02 /= a0;
    BiquadFilter(outWavSamples, a1, a2, b02, b1, b02);
    return outWavSamples;
}

// High-pass filter, same concept
std::vector<float> HighPassFilter(
    const std::vector<float>& wavSamples,
    float freqCutoff,
    UINT sampleRate,
    float q=0.70710678
) {
    std::vector<float> outWavSamples = wavSamples;

    float w0 = 2.0f * 3.1415926f * freqCutoff / sampleRate;
    float cosw0 = std::cos(w0);
    float sinw0 = std::sin(w0);
    float alpha0 = sinw0 * q / 2;

    float a0 = 1.0f + alpha0;
    float a1 = -2.0f * cosw0;
    float a2 = 1.0f - alpha0;
    float b1 = -(1.0f + cosw0);
    float b02 = -b1 / 2;
	a1 /= a0, a2 /= a0, b1 /= a0, b02 /= a0;
    BiquadFilter(outWavSamples, a1, a2, b02, b1, b02);
    return outWavSamples;
}

// Band-pass filter
std::vector<float> BandPassFilter(
    const std::vector<float>& wavSamples,
    float freqCutoff,
    UINT sampleRate,
    float q=1.0f
) {
    std::vector<float> outWavSamples = wavSamples;

    float w0 = 2.0f * 3.1415926f * freqCutoff / sampleRate;
    float cosw0 = std::cos(w0);
    float sinw0 = std::sin(w0);
    float alpha = sinw0 * q / 2;

    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cosw0;
    float a2 = 1.0f - alpha;
    float b0n2 = sinw0 / 2;
    a1 /= a0, a2 /= a0, b0n2 /= a0;
    BiquadFilter(outWavSamples, a1, a2, b0n2, 0, -b0n2);
    return outWavSamples;
}

std::vector<float> CombineMono(const std::vector<float>& wavSamples1, const std::vector<float>& wavSamples2) {
    std::vector<float> outWavSamples;
    size_t len = min(wavSamples1.size(), wavSamples2.size());
    outWavSamples.resize(len);
    for (size_t i = 0; i < len; i++) {
        outWavSamples[i] = (wavSamples1[i] + wavSamples2[i]) / 2;
    }
    return outWavSamples;
}

void GenerateLFEAndDistortion(
    const std::vector<float>& wavSamples1,
    const std::vector<float>& wavSamples2,
    std::vector<float>& outWavSamples,
    std::vector<float>& outWavSamplesSaturated,
    float sbFreqCutoff,
    float satFreqCutoff,
    UINT sampleRate,
    float ubDrive=2.0f
) {
    outWavSamples = CombineMono(wavSamples1, wavSamples2);
    outWavSamplesSaturated = outWavSamples;

    outWavSamples = LowPassFilter(outWavSamples, sbFreqCutoff, sampleRate);
	for (size_t i = 0; i < outWavSamplesSaturated.size(); i++) {
        outWavSamplesSaturated[i] -= outWavSamples[i];
    }
    SoftSaturator(outWavSamplesSaturated, ubDrive);
    outWavSamplesSaturated = LowPassFilter(outWavSamplesSaturated, satFreqCutoff, sampleRate);
}

void GenerateMRHAndSaturation(
	const std::vector<float>& wavSamples,
	std::vector<float>& outWavSamples,
	std::vector<float>& outWavSamplesSaturated,
    float sbFreqCutoff,
    float satFreqCutoff,
    UINT sampleRate,
    float ubDrive = 2.0f
) {
    outWavSamples = HighPassFilter(wavSamples, sbFreqCutoff, sampleRate);
    outWavSamplesSaturated = outWavSamples;
    SoftSaturator(outWavSamplesSaturated, ubDrive);
    outWavSamplesSaturated = LowPassFilter(outWavSamplesSaturated, satFreqCutoff, sampleRate);
}

// COM release utility
template <class T> void SafeRelease(T** ppT)
{
    if (*ppT)
    {
        (*ppT)->Release();
        *ppT = NULL;
    }
}

// Resampler using Windows Media Foundation, assisted by Copilot
HRESULT ResampleAudio(std::vector<float>& wavSamples, UINT sampleRate, UINT targetRate, HRESULT& hr) {
    IMFTransform* pResampler = nullptr;
    IMFMediaType* pInType = nullptr;
    IMFMediaType* pOutType = nullptr;
    IMFSample* pSample = nullptr;
    IMFMediaBuffer* pBuffer = nullptr;

    size_t inBytes;
    BYTE* pData = nullptr;
    DWORD cbMax = 0;
    DWORD cbCurr = 0;

    std::vector<float> outWavSamples;

    UINT32 frameSize = sizeof(float);
    UINT32 bytesPerSecond;

    MFT_OUTPUT_STREAM_INFO streamInfo = {};
    DWORD status = 0;
    MFT_OUTPUT_DATA_BUFFER outBuffer = {};

    if (sampleRate == targetRate) goto done;

    hr = CoCreateInstance(CLSID_AudioResamplerMediaObject, nullptr, CLSCTX_ALL, IID_PPV_ARGS(&pResampler));
    if (FAILED(hr)) goto done;

    hr = MFCreateMediaType(&pInType);
    hr = pInType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
    hr = pInType->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float);
    hr = pInType->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, 1);
    hr = pInType->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, sampleRate);
    hr = pInType->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32);
    hr = pInType->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, frameSize);
    bytesPerSecond = frameSize * static_cast<UINT32>(sampleRate);
    hr = pInType->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, bytesPerSecond);
    if (FAILED(hr)) goto done;

    hr = pResampler->SetInputType(0, pInType, 0);
    if (FAILED(hr)) goto done;

    hr = MFCreateMediaType(&pOutType);
    hr = pOutType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
    hr = pOutType->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float);
    hr = pOutType->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, 1);
    hr = pOutType->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, targetRate);
    hr = pOutType->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32);
    hr = pOutType->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, sizeof(float));
    hr = pOutType->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, frameSize);
    bytesPerSecond = frameSize * static_cast<UINT32>(targetRate);
    hr = pOutType->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, bytesPerSecond);
    if (FAILED(hr)) goto done;


    hr = pResampler->SetOutputType(0, pOutType, 0);
    if (FAILED(hr)) goto done;

    // Prepare a single input sample containing all input frames (one-shot)
    inBytes = wavSamples.size() * sizeof(float);
    if (inBytes == 0) { hr = S_OK; goto done; }

    hr = MFCreateMemoryBuffer(static_cast<DWORD>(inBytes), &pBuffer);
    if (FAILED(hr)) goto done;

    // Copy samples into buffer
    hr = pBuffer->Lock(&pData, &cbMax, &cbCurr);
    if (FAILED(hr)) goto done;
    memcpy(pData, wavSamples.data(), inBytes);
    hr = pBuffer->SetCurrentLength(static_cast<DWORD>(inBytes));
    pBuffer->Unlock();
    if (FAILED(hr)) goto done;

    hr = MFCreateSample(&pSample);
    if (FAILED(hr)) goto done;
    hr = pSample->AddBuffer(pBuffer);
    if (FAILED(hr)) goto done;

    // Feed the input to the resampler
    hr = pResampler->ProcessInput(0, pSample, 0);
    if (FAILED(hr)) goto done;

    // Pull output; ProcessOutput may return multiple output samples.

    while (true) {
        IMFSample* pOutSample = nullptr;
        IMFMediaBuffer* pOutMediaBuffer = nullptr;

        hr = MFCreateSample(&pOutSample);
        hr = MFCreateMemoryBuffer(static_cast<DWORD>(inBytes), &pOutMediaBuffer);
        hr = pOutSample->AddBuffer(pOutMediaBuffer);
        outBuffer.pSample = pOutSample;
        hr = pResampler->ProcessOutput(0, 1, &outBuffer, &status);
        if (hr == MF_E_TRANSFORM_NEED_MORE_INPUT) {
            hr = S_OK;
            break;
        }
        if (FAILED(hr)) break;

        pOutSample = outBuffer.pSample;
        if (pOutSample) {
            hr = pOutSample->ConvertToContiguousBuffer(&pOutMediaBuffer);
            if (SUCCEEDED(hr) && pOutMediaBuffer) {
                BYTE* pOutData = nullptr;
                DWORD cbMax = 0;
                DWORD cbCurr = 0;
                hr = pOutMediaBuffer->Lock(&pOutData, &cbMax, &cbCurr);
                if (SUCCEEDED(hr)) {
                    size_t outFloats = cbCurr / sizeof(float);
                    size_t prev = outWavSamples.size();
                    outWavSamples.resize(prev + outFloats);
                    memcpy(outWavSamples.data() + prev, pOutData, cbCurr);
                    pOutMediaBuffer->Unlock();
                }
                SafeRelease(&pOutMediaBuffer);
            }
            SafeRelease(&pOutSample);
        }
    }

    wavSamples.swap(outWavSamples);

done:
    SafeRelease(&pSample);
    SafeRelease(&pBuffer);
    SafeRelease(&pInType);
    SafeRelease(&pOutType);
    SafeRelease(&pResampler);
    return hr;
}

// Audio file read function using Windows Media Foundation, from https://learn.microsoft.com/en-us/windows/win32/medfound/tutorial--decoding-audio
HRESULT ReadAudioFile(const wchar_t* source, std::vector<float>& leftWavSamples, std::vector<float>& rightWavSamples, UINT& sampleRate, HRESULT& hr) {
	IMFSourceReader* pReader = nullptr;
    IMFMediaType* pPartialType = nullptr;
    IMFMediaType* pAudioType = nullptr;    // Represents the PCM audio format.

    DWORD cbAudioData = 0;
    DWORD cbBuffer = 0;
    BYTE* pAudioData = nullptr;
	IMFSample* pSample = nullptr;
    IMFMediaBuffer* pBuffer = nullptr;

    UINT32 channelsAttr;
    UINT32 sampleRateAttr;

    hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) {
        std::wcerr << "Failed to initialize Media Foundation." << std::endl;
        goto done;
    }

    hr = MFCreateSourceReaderFromURL(source, nullptr, &pReader);
    if (FAILED(hr)) {
        std::wcerr << "Error opening file: " << source << std::endl;
        goto done;
    }

    // Select the first audio stream, and deselect all other streams.
    hr = pReader->SetStreamSelection((DWORD)MF_SOURCE_READER_ALL_STREAMS, FALSE);
    hr = pReader->SetStreamSelection((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, TRUE);

    // Create a partial media type that specifies 32-bit float PCM.
    hr = MFCreateMediaType(&pPartialType);
    hr = pPartialType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
    hr = pPartialType->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float);

    hr = pReader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, pPartialType);

    // Get the complete uncompressed format.
    hr = pReader->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, &pAudioType);

    // Ensure the stream is selected.
    hr = pReader->SetStreamSelection((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, TRUE);

    // Get sample rate and check if stereo
    channelsAttr = MFGetAttributeUINT32(pAudioType, MF_MT_AUDIO_NUM_CHANNELS, 0);
    sampleRateAttr = MFGetAttributeUINT32(pAudioType, MF_MT_AUDIO_SAMPLES_PER_SECOND, 0);
    if (channelsAttr != 2) {
        std::wcerr << "Audio is not stereo." << std::endl;
        hr = E_FAIL;
        goto done;
    }
    sampleRate = static_cast<UINT>(sampleRateAttr);

    // Return the PCM format to the caller.
    pAudioType->AddRef();

    // Read samples
    leftWavSamples.clear();
    rightWavSamples.clear();

    while (true) {
        DWORD dwFlags = 0;

        hr = pReader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, nullptr, &dwFlags, nullptr, &pSample);

        if (FAILED(hr)) break;
        if (dwFlags & MF_SOURCE_READERF_ENDOFSTREAM) {
            hr = S_OK;
            break;
        }
        if (pSample == nullptr) {
            std::wcerr << "No sample." << std::endl;
            continue;
        }

        // Get a pointer to the audio data in the sample.
        hr = pSample->ConvertToContiguousBuffer(&pBuffer);
        if (FAILED(hr)) break;
        hr = pBuffer->Lock(&pAudioData, nullptr, &cbBuffer);
        if (FAILED(hr)) break;

        // Interpret channels as interleaved (Copilot told me this, confirmed at https://docs.omniverse.nvidia.com/kit/docs/carbonite/158.6/docs/audio/Basics.html)
        size_t frameCount = cbBuffer / (sizeof(float) * 2);
        size_t prevSize = leftWavSamples.size();
        float* rawWavSamples = reinterpret_cast<float*>(pAudioData);
        leftWavSamples.resize(prevSize + frameCount);
        rightWavSamples.resize(prevSize + frameCount);


        for (size_t i = 0; i < frameCount; i++) {
            leftWavSamples[prevSize + i] = rawWavSamples[(i << 1)];
            rightWavSamples[prevSize + i] = rawWavSamples[(i << 1) | size_t(1)];
        }

        // Unlock the buffer.
        hr = pBuffer->Unlock();
        pAudioData = nullptr;
        SafeRelease(&pSample);
        SafeRelease(&pBuffer);
    }

    if (pAudioData) {
        pBuffer->Unlock();
    }

done:
    SafeRelease(&pPartialType);
    SafeRelease(&pBuffer);
    SafeRelease(&pSample);
    SafeRelease(&pAudioType);
    SafeRelease(&pReader);

    return hr;
}

void SetupAudioEnvironment(
    UINT targetRate,
    std::vector<Windows::Foundation::Numerics::float3>& positions,
    std::vector<size_t>& channels,
    std::vector<float>& volumes,
    std::vector<UINT>& offsets
) {
    float masterVolume = 0.5;

    positions = {
        Windows::Foundation::Numerics::float3(-0.5f, -0.5f, -1.0f),
        Windows::Foundation::Numerics::float3(-0.5f, -0.5f, 1.0f),

        Windows::Foundation::Numerics::float3(1.0f, -0.5f, -1.0f),
        Windows::Foundation::Numerics::float3(1.0f, -0.5f, 1.0f),

        Windows::Foundation::Numerics::float3(0.5f, 0.0f, 0.0f),

        Windows::Foundation::Numerics::float3(-0.5f, -0.5f, 2.0f),
        Windows::Foundation::Numerics::float3(0.0f, 0.0f, 0.0f),
        Windows::Foundation::Numerics::float3(1.0f, 0.0f, 0.0f),
        Windows::Foundation::Numerics::float3(0.5f, 0.0f, 1.0f),
    };
    channels = {
        SpeakerChannel_LeftMRH,
        SpeakerChannel_LeftMRH,

        SpeakerChannel_RightMRH,
        SpeakerChannel_RightMRH,

        SpeakerChannel_MonoDistortion,

        SpeakerChannel_MonoLFE,
        SpeakerChannel_MonoLFE,
        SpeakerChannel_MonoLFE,
        SpeakerChannel_MonoLFE,
    };
    volumes = {
        0.25f,
        0.15f,

        0.25f,
        0.15f,

        0.6f,

        0.8f,
        0.4f,
        0.4f,
        0.2f,
    };
    std::vector<float> delays = {
        0.0f,
        0.0f,

        0.0f,
        0.0f,

        2.0f,

        0.0f,
        2.0f,
        2.0f,
        4.0f,
    }; // In ms

    // Offsets for sound travel
    float speedOfSound = 100.0f; // In m/s (more realistically, the "how-wide-o-stat")

    offsets.resize(positions.size());

    float samplesPerMeter = static_cast<float>(targetRate) / speedOfSound;
    float samplesPerMillisecond = static_cast<float>(targetRate) / 1000.0f;
    UINT maxOffset = 0;
    for (size_t i = 0; i < positions.size(); i++) {
        offsets[i] = static_cast<UINT>((Windows::Foundation::Numerics::length(positions[i]) * samplesPerMeter) + (delays[i] * samplesPerMillisecond));
        if (offsets[i] > maxOffset) {
            maxOffset = offsets[i];
        }
    }
    for (size_t i = 0; i < offsets.size(); i++) {
        offsets[i] = maxOffset - offsets[i];
    }

    // Applying master volume
    for (float& volume : volumes) {
        volume *= masterVolume;
    }
}

static HRESULT SetupAudioDevices(
    Microsoft::WRL::ComPtr<ISpatialAudioObjectRenderStream>& spatialAudioStream,
    HANDLE& bufferCompletionEvent,
    UINT targetRate,
    HRESULT& hr
) {

    // This is where we get the audio device
    Microsoft::WRL::ComPtr<IMMDeviceEnumerator> deviceEnum;
    Microsoft::WRL::ComPtr<IMMDevice> defaultDevice;

    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&deviceEnum);
    hr = deviceEnum->GetDefaultAudioEndpoint(EDataFlow::eRender, eMultimedia, &defaultDevice);

    // This is where the audio format is specified (32-bit, mono, 48kHz, likely according to system settings)
    WAVEFORMATEX format;
    format.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
    format.wBitsPerSample = 32;
    format.nChannels = 1;
    format.nSamplesPerSec = targetRate;
    format.nBlockAlign = (format.wBitsPerSample >> 3) * format.nChannels;
    format.nAvgBytesPerSec = format.nBlockAlign * format.nSamplesPerSec;
    format.cbSize = 0;

    // Activate ISpatialAudioClient on the desired audio-device 
    Microsoft::WRL::ComPtr<ISpatialAudioClient> spatialAudioClient;
    hr = defaultDevice->Activate(__uuidof(ISpatialAudioClient), CLSCTX_INPROC_SERVER, nullptr, (void**)&spatialAudioClient);

    hr = spatialAudioClient->IsAudioObjectFormatSupported(&format);
    if (FAILED(hr)) {
        std::wcerr << "Audio object format not supported by spatial audio client." << std::endl;
        return hr;
    }

    // Create the event that will be used to signal the client for more data
    bufferCompletionEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // 220 max dynamic objects for Windows Sonic UWP
    UINT32 maxDynamicObjectCount;
    hr = spatialAudioClient->GetMaxDynamicObjectCount(&maxDynamicObjectCount);

    if (maxDynamicObjectCount == 0) {
        // Dynamic objects are unsupported
        return hr;
    }

    SpatialAudioObjectRenderStreamActivationParams streamParams;
    streamParams.ObjectFormat = &format;
    streamParams.StaticObjectTypeMask = AudioObjectType_None;
    streamParams.MinDynamicObjectCount = 0;
    streamParams.MaxDynamicObjectCount = min(maxDynamicObjectCount, 20);
    streamParams.Category = AudioCategory_GameEffects;
    streamParams.EventHandle = bufferCompletionEvent;
    streamParams.NotifyObject = nullptr;

    PROPVARIANT pv;
    PropVariantInit(&pv);
    pv.vt = VT_BLOB;
    pv.blob.cbSize = sizeof(streamParams);
    pv.blob.pBlobData = (BYTE*)&streamParams;

    hr = spatialAudioClient->ActivateSpatialAudioStream(&pv, __uuidof(spatialAudioStream), (void**)&spatialAudioStream);

    return hr;
}

// Spatial audio streaming function
static HRESULT StreamSpatialAudio(
    Microsoft::WRL::ComPtr<ISpatialAudioObjectRenderStream>& spatialAudioStream,
    HANDLE& bufferCompletionEvent,
    std::vector<std::vector<float>>& wavSamples,
    std::vector<Windows::Foundation::Numerics::float3>& positions,
    std::vector<size_t>& channels,
    std::vector<float> volumes,
    std::vector<UINT> offsets,
    HRESULT& hr
) {
    // Initializing sound object
    std::vector<Speaker3dObject> speakerObjects;
    for (size_t i = 0; i < positions.size(); i++) {
        Microsoft::WRL::ComPtr<ISpatialAudioObject> audioObject;
        hr = spatialAudioStream->ActivateSpatialAudioObject(AudioObjectType::AudioObjectType_Dynamic, &audioObject);

        speakerObjects.push_back({
            audioObject,
            positions.at(i),
            volumes.at(i),
            &wavSamples.at(channels.at(i)),
            static_cast<UINT>(wavSamples.at(channels.at(i)).size()),
            offsets.at(i)
        });
    }

    std::wcout << "Beginning spatial audio streaming." << std::endl;

    // Start streaming / rendering
    hr = spatialAudioStream->Start();

    float rotateAngle = 0.0f;
    bool streaming = true;
    while (streaming) {
        // Wait for a signal from the audio-engine to start the next processing pass
        if (WaitForSingleObject(bufferCompletionEvent, 200) != WAIT_OBJECT_0) {
            break;
        }

        UINT32 availableDynamicObjectCount;
        UINT32 frameCount;

        hr = spatialAudioStream->BeginUpdatingAudioObjects(&availableDynamicObjectCount, &frameCount);

        BYTE* buffer;
        UINT32 bufferLength;

        std::vector<Speaker3dObject>::iterator it = speakerObjects.begin();
        while (it != speakerObjects.end()) {
            hr = it->audioObject->GetBuffer(&buffer, &bufferLength);
            UINT writeLen = 0;

            // End of data
            if (it->totalFrameCount > it->currFrameIndex) {
                // Write audio to available frames
                writeLen = WriteToAudioObjectBuffer(reinterpret_cast<float*>(buffer), frameCount, *(it->wavSample), it->currFrameIndex);

                // Audio object has to be updated for each iteration, will return to default value if only set once
                hr = it->audioObject->SetPosition(it->position.x, it->position.y, it->position.z);
                hr = it->audioObject->SetVolume(it->volume);

                it++;
            }
            else {
                hr = it->audioObject->SetEndOfStream(writeLen);
                it->audioObject = nullptr;
                it->totalFrameCount = 0;
                it = speakerObjects.erase(it);
                streaming = false;
            }
        }

        hr = spatialAudioStream->EndUpdatingAudioObjects();
    }

    // Stop the stream 
    hr = spatialAudioStream->Stop();

    // We don't want to start again, so reset the stream to free it's resources.
    hr = spatialAudioStream->Reset();

    CloseHandle(bufferCompletionEvent);

    return hr;
}

int main(int argc, char* argv[]) {
    HeapSetInformation(NULL, HeapEnableTerminationOnCorruption, NULL, 0);

    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);

    if (FAILED(hr)) {
        std::wcerr << "Failed to initialize COM library." << std::endl;
        return 1;
    }

    Microsoft::WRL::ComPtr<ISpatialAudioObjectRenderStream> spatialAudioStream;
    HANDLE bufferCompletionEvent;
    UINT targetRate = 48000;
    std::vector<Windows::Foundation::Numerics::float3> positions;
    std::vector<size_t> channels;
    std::vector<float> volumes;
    std::vector<UINT> offsets;
    SetupAudioEnvironment(targetRate, positions, channels, volumes, offsets);

    // For non-Latin characters
    if (_setmode(_fileno(stdin), _O_U16TEXT) == -1) {
        std::wcerr << "stdin UTF16 mode failed." << std::endl;
        return 1;
    }
    if (_setmode(_fileno(stdout), _O_U16TEXT) == -1) {
        std::wcerr << "stdout UTF16 mode failed." << std::endl;
        return 1;
    }

    // Getting file
    std::wcout << "Path to folder (in quotations): ";
    std::wstring dirPath;
	std::getline(std::wcin, dirPath);
    dirPath.pop_back();
    dirPath.erase(0, 1);
    std::wcout << std::endl;
    std::vector<std::wstring> wavPaths;


    if (std::filesystem::is_directory(dirPath)) {
        std::wcout << "Loading folder: " << dirPath << std::endl;
        for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
            if (std::filesystem::exists(entry.path()) && std::filesystem::is_regular_file(entry.path())) {
                wavPaths.push_back(entry.path());
            }
        }
        UINT seed = (UINT)std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 gen(seed);
        std::shuffle(std::begin(wavPaths), std::end(wavPaths), gen);
    }
    else {
        wavPaths.push_back(dirPath);
    }

    // Load WAV samples into vector<float>
	std::vector<float> leftWavSamples;
    leftWavSamples.resize(33554432);
    std::vector<float> rightWavSamples;
    rightWavSamples.resize(33554432);
    std::vector<std::vector<float>> wavSamples;
    wavSamples.resize(5);
    for (std::vector<float>& wavSample : wavSamples) {
        wavSample.resize(33554432);
    }
    UINT wavSampleRate;
    for (const std::wstring& wavPath : wavPaths) {

        SetupAudioDevices(spatialAudioStream, bufferCompletionEvent, targetRate, hr);
        if (FAILED(hr)) {
            std::wcerr << "Spatial audio setup failed." << std::endl;
            return 1;
        }

        std::wcout << "Loading file: " << wavPath << std::endl;
        hr = ReadAudioFile(wavPath.c_str(), leftWavSamples, rightWavSamples, wavSampleRate, hr);
        if (FAILED(hr)) {
            std::wcerr << "File loading failed." << std::endl;
            continue;
        }
        std::wcout << "Buffer size: " << rightWavSamples.size() << std::endl;

        // Resample to 48kHz
        // Memory management sucks here
        std::vector<std::thread> threads;
        threads.reserve(2);
        threads.emplace_back([&] {
            hr = ResampleAudio(leftWavSamples, wavSampleRate, targetRate, hr);
        });
        threads.emplace_back([&] {
            hr = ResampleAudio(rightWavSamples, wavSampleRate, targetRate, hr);
        });
        for (std::thread& t : threads) {
            t.join();
        }
        if (FAILED(hr)) {
            std::wcerr << "Audio resampling failed." << std::endl;
            continue;
        }
        std::wcout << "Buffer size (resampled): " << rightWavSamples.size() << std::endl;

        threads.clear();
        threads.reserve(3);
        threads.emplace_back([&] {
		    GenerateLFEAndDistortion(leftWavSamples, rightWavSamples, wavSamples[SpeakerChannel_MonoLFE], wavSamples[SpeakerChannel_MonoDistortion], 80.0f, 100.0f, wavSampleRate, 2.8f);
        });
        threads.emplace_back([&] {
            wavSamples[SpeakerChannel_LeftMRH] = HighPassFilter(leftWavSamples, 80.0f, wavSampleRate);
        });
        threads.emplace_back([&] {
            wavSamples[SpeakerChannel_RightMRH] = HighPassFilter(rightWavSamples, 80.0f, wavSampleRate);
        });
        for (std::thread& t : threads) {
            t.join();
        }
        leftWavSamples.clear();
        rightWavSamples.clear();

        hr = StreamSpatialAudio(spatialAudioStream, bufferCompletionEvent, wavSamples, positions, channels, volumes, offsets, hr);

        for (std::vector<float>& wavSample : wavSamples) {
            wavSample.clear();
        }

        if (FAILED(hr)) {
            std::wcerr << "Spatial audio streaming failed." << std::endl;
            return 1;
	    }
    }

    return 0;
}