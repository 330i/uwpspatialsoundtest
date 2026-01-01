// uwpspatialsoundtest.cpp : This file contains the 'main' function. Program execution begins and ends there.
// Most (if not all) of these are from https://learn.microsoft.com/en-us/windows/win32/coreaudio/render-spatial-sound-using-spatial-audio-objects

#include <iostream>
#include <fstream>
#include <random>
#include <stdlib.h>
#include <locale.h>
#include <thread>

#include <windows.foundation.h>
#include <windowsnumerics.h>
#include <wrl/wrappers/corewrappers.h>
#include <wrl/client.h>
#include <SpatialAudioClient.h>
#include <mmdeviceapi.h>

enum FilterType {
    FilterType_LowPass,
    FilterType_HighPass,
    FilterTYpe_LFE
};

// Audio object struct
struct Speaker3dObject {
    Microsoft::WRL::ComPtr<ISpatialAudioObject> audioObject;
	Windows::Foundation::Numerics::float3 position; // In meters
    float volume; // 0.0 to 1.0
    std::vector<float>* wavSample;
    UINT totalFrameCount;
    UINT currFrameIndex;   // Since chunks of audio are written to buffer
};

// Writes frameCount samples from PCM sample vector to buffer
UINT WriteToAudioObjectBuffer(FLOAT* buffer, UINT frameCount, const std::vector<float> &data, UINT &pos) {
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

inline void BiquadFilter(std::vector<float>& wavSamples, float a1, float a2, float b02, float b1) {
    float y = 0.0f;
    float prevTerm = 0.0f;
    float prevPrevTerm = 0.0f;
    std::vector<float>::iterator it = wavSamples.begin();
    while (it != wavSamples.end()) {
        y = b02 * (*it) + prevTerm;
        prevTerm = b1 * (*it) - a1 * y + prevPrevTerm;
        prevPrevTerm = b02 * (*it) - a2 * y;
        *it = y;
        it++;
    }
}

inline void OnePole(std::vector<float>& wavSamples, float freqCutoff, UINT sampleRate) {
    float a = std::exp(-2.0f * 3.1415926f * freqCutoff / sampleRate);
    float b = 1.0f - a;
    float prevY = 0.0f;
    std::vector<float>::iterator it = wavSamples.begin();
    while (it != wavSamples.end()) {
        *it = b * (*it) + a * prevY;
        prevY = *it;
        it++;
    }
}

// Soft saturator for "boominess"
inline void SoftSaturator(std::vector<float>& wavSamples, float drive) {
    std::vector<float>::iterator it = wavSamples.begin();
    while (it != wavSamples.end()) {
        *it = std::tanh(drive * (*it));
        it++;
    }
}

// Low-pass filter, uses two second-order IIR filters
void LowPassFilter(const std::vector<float> &wavSamples, std::vector<float>*& outWavSamplesPtr, float freqCutoff, UINT sampleRate) {
    outWavSamplesPtr = new std::vector<float>(wavSamples);

    // Approximation of w0 from https://dsp.stackexchange.com/questions/28308/3-db-cut-off-frequency-of-exponentially-weighted-moving-average-filter/28314#28314
    // Biquad filter from https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
	// The Q values are from 4th-order Butterworth low-pass Perplexity got that I don't feel qualified to understand
    float w0 = 2.0f * 3.1415926f * freqCutoff / sampleRate;
	float cosw0 = std::cos(w0);
	float sinw0 = std::sin(w0);
    float alpha0 = sinw0 * 0.54119611f / 2;
	float alpha1 = sinw0 * 1.30656296f / 2;

	float a0 = 1.0f + alpha0;
	float a1 = -2.0f * cosw0;
	float a2 = 1.0f - alpha0;
    float b1 = 1.0f - cosw0;
	float b02 = b1 / 2;
    BiquadFilter(*outWavSamplesPtr, a1 / a0, a2 / a0, b02 / a0, b1 / a0);

    a0 = 1.0f + alpha1;
    a2 = 1.0f - alpha1;
    BiquadFilter(*outWavSamplesPtr, a1 / a0, a2 / a0, b02 / a0, b1 / a0);
}

// High-pass filter, same concept
void HighPassFilter(const std::vector<float>& wavSamples, std::vector<float>*& outWavSamplesPtr, float freqCutoff, UINT sampleRate) {
    outWavSamplesPtr = new std::vector<float>(wavSamples);

    float w0 = 2.0f * 3.1415926f * freqCutoff / sampleRate;
    float cosw0 = std::cos(w0);
    float sinw0 = std::sin(w0);
    float alpha0 = sinw0 * 0.54119611f / 2;
    float alpha1 = sinw0 * 1.30656296f / 2;

    float a0 = 1.0f + alpha0;
    float a1 = -2.0f * cosw0;
    float a2 = 1.0f - alpha0;
    float b1 = -(1.0f + cosw0);
    float b02 = -b1 / 2;
    BiquadFilter(*outWavSamplesPtr, a1 / a0, a2 / a0, b02 / a0, b1 / a0);

    a0 = 1.0f + alpha1;
    a2 = 1.0f - alpha1;
    BiquadFilter(*outWavSamplesPtr, a1 / a0, a2 / a0, b02 / a0, b1 / a0);
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

void LFEGenerator(const std::vector<float>& wavSamples1, const std::vector<float>& wavSamples2, std::vector<float>*& outWavSamplesPtr, UINT sampleRate, float sbGain=1.0f, float ubGain=0.5f, float ubDrive=2.0f) {
    outWavSamplesPtr = new std::vector<float>(CombineMono(wavSamples1, wavSamples2));
    std::vector<float>* sbWavSamples = nullptr;
    std::vector<float>* ubWavSamples = nullptr;

    std::vector<std::thread> threads;
    threads.reserve(2);
    threads.emplace_back([&] {
        LowPassFilter(*outWavSamplesPtr, sbWavSamples, 80.0f, sampleRate);
    });
    threads.emplace_back([&] {
        LowPassFilter(*outWavSamplesPtr, ubWavSamples, 200.0f, sampleRate);
    });

    for (std::thread& t : threads) {
        t.join();
    }
    threads.clear();

    threads.emplace_back([&] {
        HighPassFilter(*ubWavSamples, ubWavSamples, 80.0f, sampleRate);
    });
    threads.emplace_back([&] {
        SoftSaturator(*ubWavSamples, ubDrive);
    });

    for (std::thread& t : threads) {
        t.join();
    }

    for (size_t i = 0; i < (*outWavSamplesPtr).size(); i++) {
        outWavSamplesPtr->at(i) = sbGain * sbWavSamples->at(i) + ubGain * ubWavSamples->at(i);
    }
}


// These 24-bit WAV loading utility functions are generated by Copilot.
// TODO: Change them to dynamically load other bit depths, maybe use avlibcodec (?)

// Convert signed 24-bit little-endian (3 bytes) to int32 with sign extension
static inline int32_t Int24ToInt32(const uint8_t bytes[3])
{
    int32_t val = (bytes[0]) | (bytes[1] << 8) | (bytes[2] << 16);
    // sign extend if negative (bit 23 set)
    if (val & 0x00800000) {
        val |= 0xFF000000;
    }
    return val;
}

// Load a mono 24-bit PCM WAV file and convert to float32 samples in range [-1,1).
// Returns true on success and fills outSamples and outSampleRate.
bool LoadWav24ToFloat(const std::wstring& path, std::vector<float>& outSamples, UINT& outSampleRate)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    // Read RIFF header
    char riff[4];
    uint32_t riffChunkSize = 0;
    char wave[4];
    ifs.read(riff, 4);
    ifs.read(reinterpret_cast<char*>(&riffChunkSize), 4);
    ifs.read(wave, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0)
        return false;

    // Parse chunks
    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
    uint16_t bitsPerSample = 0;
    std::vector<uint8_t> dataBytes;

    while (ifs && !ifs.eof()) {
        char chunkId[4];
        uint32_t chunkSize = 0;
        ifs.read(chunkId, 4);
        if (!ifs) break;
        ifs.read(reinterpret_cast<char*>(&chunkSize), 4);
        if (!ifs) break;

        std::streampos nextChunk = ifs.tellg();
        nextChunk += static_cast<std::streamoff>(chunkSize);

        if (std::strncmp(chunkId, "fmt ", 4) == 0) {
            // Read format chunk (at least 16 bytes)
            if (chunkSize < 16) return false;
            ifs.read(reinterpret_cast<char*>(&audioFormat), sizeof(audioFormat));
            ifs.read(reinterpret_cast<char*>(&numChannels), sizeof(numChannels));
            ifs.read(reinterpret_cast<char*>(&sampleRate), sizeof(sampleRate));
            uint32_t byteRate = 0;
            uint16_t blockAlign = 0;
            ifs.read(reinterpret_cast<char*>(&byteRate), sizeof(byteRate));
            ifs.read(reinterpret_cast<char*>(&blockAlign), sizeof(blockAlign));
            ifs.read(reinterpret_cast<char*>(&bitsPerSample), sizeof(bitsPerSample));
            // skip any extra fmt bytes
        }
        else if (std::strncmp(chunkId, "data", 4) == 0) {
            dataBytes.resize(chunkSize);
            ifs.read(reinterpret_cast<char*>(dataBytes.data()), chunkSize);
        }
        // advance to next chunk (handle odd chunk sizes)
        ifs.seekg(nextChunk);
    }

    if (audioFormat != 1) return false; // expect PCM (1)
    if (numChannels != 1) return false;  // expect mono
    if (bitsPerSample != 24) return false; // expect 24-bit

    // Convert 3-byte samples to float
    const size_t bytesPerSample = 3;
    size_t sampleCount = dataBytes.size() / bytesPerSample;
    outSamples.resize(sampleCount);
    const float denom = 8388608.0f; // 2^23
    for(size_t i = 0; i < sampleCount; ++i) {
        const uint8_t* b = &dataBytes[i * bytesPerSample];
        int32_t s = Int24ToInt32(b);
        outSamples[i] = static_cast<float>(s) / denom;
    }

    outSampleRate = sampleRate;
    return true;
}

// Linear resampler from inRate -> outRate.
// Simple, fast, acceptable for small test/demo. Use a higher-quality library for production.
void ResampleLinear(const std::vector<float>& in, UINT inRate, UINT outRate, std::vector<float>& out)
{
    if (inRate == outRate) {
        out = in;
        return;
    }
    if (in.empty()) {
        out.clear();
        return;
    }

    double ratio = static_cast<double>(outRate) / static_cast<double>(inRate);
    size_t outLen = static_cast<size_t>(std::floor(in.size() * ratio + 0.5));
    out.resize(outLen);

    for (size_t j = 0; j < outLen; ++j) {
        double srcPos = static_cast<double>(j) / ratio;
        size_t i0 = static_cast<size_t>(std::floor(srcPos));
        double frac = srcPos - static_cast<double>(i0);
        float s0 = in[min(i0, in.size() - 1)];
        float s1 = in[min(i0 + 1, in.size() - 1)];
        out[j] = static_cast<float>((1.0 - frac) * s0 + frac * s1);
    }
}



int main(int argc, char* argv[]) {
    // Getting WAV file (test file uses 24-bit, mono, 44.100kHz at the moment)
    if (argc < 3) {
		std::cerr << "Need a path to a mono 24-bit WAV file as an argument, separate LR channels." << std::endl;
        return 1;
    }
	
	std::setlocale(LC_ALL, ""); // Default locale
	size_t requiredSize = strlen(argv[1]) + 1;
	wchar_t* leftPwc = new wchar_t[requiredSize];
    size_t outWSize;
	mbstowcs_s(&outWSize, leftPwc, requiredSize, argv[1], requiredSize - 1);
    std::wstring leftWavPath(leftPwc);
    delete[] leftPwc;

    requiredSize = strlen(argv[2]) + 1;
    wchar_t* rightPwc = new wchar_t[requiredSize];
    mbstowcs_s(&outWSize, rightPwc, requiredSize, argv[2], requiredSize - 1);
    std::wstring rightWavPath(rightPwc);
    delete[] rightPwc;

    std::wcout << "Loading WAV files: " << leftWavPath << " (left), " << rightWavPath << " (right)" << std::endl;

    // Load WAV samples into vector<float>
	std::vector<float> leftWavSamples;
	UINT wavSampleRate;
	if (!LoadWav24ToFloat(leftWavPath, leftWavSamples, wavSampleRate)) {
        std::cerr << "Failed to load WAV (left)." << std::endl;
        return 1;
	}
    std::vector<float> rightWavSamples;
    if (!LoadWav24ToFloat(rightWavPath, rightWavSamples, wavSampleRate)) {
        std::cerr << "Failed to load WAV (right)." << std::endl;
        return 1;
    }

    std::vector<float>* monoWavSamplesLFE = nullptr;
    std::vector<float>* leftWavSamplesMRH = nullptr;
    std::vector<float>* rightWavSamplesMRH = nullptr;

    std::vector<std::thread> threads;
    threads.reserve(3);
    threads.emplace_back([&] {
        LFEGenerator(leftWavSamples, rightWavSamples, monoWavSamplesLFE, wavSampleRate, 1.0f, 0.6f, 2.0f);
    });
    threads.emplace_back([&] {
        HighPassFilter(leftWavSamples, leftWavSamplesMRH, 200.0f, wavSampleRate);
    });
    threads.emplace_back([&] {
        HighPassFilter(rightWavSamples, rightWavSamplesMRH, 200.0f, wavSampleRate);
    });

    for (std::thread& t : threads) {
        t.join();
    }

    // Resample to 48kHz
    const UINT targetRate = 48000;
    if (wavSampleRate != targetRate) {
        std::vector<float> resampled;

        ResampleLinear(*leftWavSamplesMRH, wavSampleRate, targetRate, resampled);
        (*leftWavSamplesMRH).swap(resampled);

        ResampleLinear(*rightWavSamplesMRH, wavSampleRate, targetRate, resampled);
        (*rightWavSamplesMRH).swap(resampled);

        ResampleLinear(*monoWavSamplesLFE, wavSampleRate, targetRate, resampled);
        (*monoWavSamplesLFE).swap(resampled);
    }


    // This is where we get the audio device
    HRESULT hr = CoInitialize(nullptr);
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

    Microsoft::WRL::ComPtr<ISpatialAudioObjectRenderStream> spatialAudioStream;

    hr = spatialAudioClient->IsAudioObjectFormatSupported(&format);
    if (FAILED(hr)) {
        std::cerr << "Audio object format not supported by spatial audio client." << std::endl;
        return 1;
    }

    // Create the event that will be used to signal the client for more data
    HANDLE bufferCompletionEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // 220 max dynamic objects for Windows Sonic UWP
    UINT32 maxDynamicObjectCount;
    hr = spatialAudioClient->GetMaxDynamicObjectCount(&maxDynamicObjectCount);

    if (maxDynamicObjectCount == 0) {
        // Dynamic objects are unsupported
        return 1;
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

    // Start streaming / rendering 
    hr = spatialAudioStream->Start();

    std::vector<std::vector<float>> wavSamples = { *leftWavSamplesMRH, *rightWavSamplesMRH, *monoWavSamplesLFE };
    std::vector<Windows::Foundation::Numerics::float3> positions = {
        Windows::Foundation::Numerics::float3(-0.5f, -1.0f, -1.0f),
        Windows::Foundation::Numerics::float3(1.0f, -1.0f, -1.0f),
        Windows::Foundation::Numerics::float3(-0.5f, 0.0f, 2.0f),
        Windows::Foundation::Numerics::float3(0.0f, 0.0f, 0.0f)
    };
    std::vector<size_t> channels = { 0, 1, 2, 2 };
    std::vector<float> volumes = { 0.5f, 0.5f, 0.7f, 0.3f };
    std::vector<Speaker3dObject> speakerObjects;
    std::vector<UINT> offsets = { 2040, 2000, 2000, 0 };

    // Initializing sound object
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

    std::cout << "Beginning spatial audio streaming." << std::endl;

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
            if (it->totalFrameCount >= it->currFrameIndex) {
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

    return 0;
}