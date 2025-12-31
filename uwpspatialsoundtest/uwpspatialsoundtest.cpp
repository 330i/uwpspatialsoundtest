// uwpspatialsoundtest.cpp : This file contains the 'main' function. Program execution begins and ends there.
// Most (if not all) of these are from https://learn.microsoft.com/en-us/windows/win32/coreaudio/render-spatial-sound-using-spatial-audio-objects

#include <iostream>
#include <fstream>
#include <random>
#include <stdlib.h>
#include <locale.h>

#include <windows.foundation.h>
#include <windowsnumerics.h>
#include <wrl/wrappers/corewrappers.h>
#include <wrl/client.h>
#include <SpatialAudioClient.h>
#include <SpatialAudioHrtf.h>
#include <mmdeviceapi.h>
#include <directxmath.h>

// Audio object struct
struct Speaker3dObject {
    Microsoft::WRL::ComPtr<ISpatialAudioObjectForHrtf> audioObject;
	Windows::Foundation::Numerics::float3 position; // In meters
    SpatialAudioHrtfOrientation* orientation;
    float gain; // In decibels
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
    float a = std::exp(-2.0f * std::_Pi_val * freqCutoff / sampleRate);
    float b = 1.0f - a;
    float prevY = 0.0f;
    std::vector<float>::iterator it = wavSamples.begin();
    while (it != wavSamples.end()) {
        *it = b * (*it) + a * prevY;
        prevY = *it;
        it++;
    }
}

// Low-pass filter, uses two second-order IIR filters
void LowPassFilter(std::vector<float> &wavSamples, float freqCutoff, UINT sampleRate) {
    // Approximation of w0 from https://dsp.stackexchange.com/questions/28308/3-db-cut-off-frequency-of-exponentially-weighted-moving-average-filter/28314#28314
    // Biquad filter from https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
	// The Q values are from 4th-order Butterworth low-pass Perplexity got that I don't feel qualified to understand
    float w0 = 2.0f * std::_Pi_val * freqCutoff / sampleRate;
	float cosw0 = std::cos(w0);
	float sinw0 = std::sin(w0);
    float alpha0 = sinw0 * 0.5f * 0.54119611f;
	float alpha1 = sinw0 * 0.5f * 1.30656296f;

	float a0 = 1.0f + alpha0;
	float a1 = -2.0f * cosw0;
	float a2 = 1.0f - alpha0;
    float b1 = 1.0f - cosw0;
	float b02 = b1 * 0.5f;
    BiquadFilter(wavSamples, a1 / a0, a2 / a0, b02 / a0, b1 / a0);

    a0 = 1.0f + alpha1;
    a2 = 1.0f - alpha1;
    BiquadFilter(wavSamples, a1 / a0, a2 / a0, b02 / a0, b1 / a0);
}

// High-pass filter, same concept
void HighPassFilter(std::vector<float>& wavSamples, float freqCutoff, UINT sampleRate) {
    float w0 = 2.0f * std::_Pi_val * freqCutoff / sampleRate;
    float cosw0 = std::cos(w0);
    float sinw0 = std::sin(w0);
    float alpha0 = sinw0 * 0.5f * 0.54119611f;
    float alpha1 = sinw0 * 0.5f * 1.30656296f;

    float a0 = 1.0f + alpha0;
    float a1 = -2.0f * cosw0;
    float a2 = 1.0f - alpha0;
    float b1 = -(1.0f + cosw0);
    float b02 = -b1 * 0.5f;
    BiquadFilter(wavSamples, a1 / a0, a2 / a0, b02 / a0, b1 / a0);

    a0 = 1.0f + alpha1;
    a2 = 1.0f - alpha1;
    BiquadFilter(wavSamples, a1 / a0, a2 / a0, b02 / a0, b1 / a0);
}

// Calculates orientation matrix, provided by Microsoft article
DirectX::XMMATRIX CalculateEmitterOrientationMatrix(Windows::Foundation::Numerics::float3 listenerOrientationFront, Windows::Foundation::Numerics::float3 emitterDirection)
{
    DirectX::XMVECTOR vListenerDirection = DirectX::XMLoadFloat3(&listenerOrientationFront);
    DirectX::XMVECTOR vEmitterDirection = DirectX::XMLoadFloat3(&emitterDirection);
    DirectX::XMVECTOR vCross = DirectX::XMVector3Cross(vListenerDirection, vEmitterDirection);
    DirectX::XMVECTOR vDot = DirectX::XMVector3Dot(vListenerDirection, vEmitterDirection);
    DirectX::XMVECTOR vAngle = DirectX::XMVectorACos(vDot);
    float angle = DirectX::XMVectorGetX(vAngle);

    // The angle must be non-zero
    if (fabsf(angle) > FLT_EPSILON)
    {
        // And less than PI
        if (fabsf(angle) < DirectX::XM_PI)
        {
            return DirectX::XMMatrixRotationAxis(vCross, angle);
        }

        // If equal to PI, find any other non-collinear vector to generate the perpendicular vector to rotate about
        else
        {
            DirectX::XMFLOAT3 vector = { 1.0f, 1.0f, 1.0f };
            if (listenerOrientationFront.x != 0.0f)
            {
                vector.x = -listenerOrientationFront.x;
            }
            else if (listenerOrientationFront.y != 0.0f)
            {
                vector.y = -listenerOrientationFront.y;
            }
            else // if (_listenerOrientationFront.z != 0.0f)
            {
                vector.z = -listenerOrientationFront.z;
            }
            DirectX::XMVECTOR vVector = DirectX::XMLoadFloat3(&vector);
            vVector = DirectX::XMVector3Normalize(vVector);
            vCross = DirectX::XMVector3Cross(vVector, vEmitterDirection);
            return DirectX::XMMatrixRotationAxis(vCross, angle);
        }
    }

    // If the angle is zero, use an identity matrix
    return DirectX::XMMatrixIdentity();
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

	//HighPassFilter(leftWavSamples, 200.0f, wavSampleRate);

    // Resample to 48kHz
    const UINT targetRate = 48000;
    if (wavSampleRate != targetRate) {
        std::vector<float> resampled;
        ResampleLinear(leftWavSamples, wavSampleRate, targetRate, resampled);
        leftWavSamples.swap(resampled);
    }


    std::vector<float> rightWavSamples;
    if (!LoadWav24ToFloat(rightWavPath, rightWavSamples, wavSampleRate)) {
        std::cerr << "Failed to load WAV (right)." << std::endl;
        return 1;
    }
    //HighPassFilter(rightWavSamples, 200.0f, wavSampleRate);
    if (wavSampleRate != targetRate) {
        std::vector<float> resampled;
        ResampleLinear(rightWavSamples, wavSampleRate, targetRate, resampled);
        rightWavSamples.swap(resampled);
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

    Microsoft::WRL::ComPtr<ISpatialAudioObjectRenderStreamForHrtf> spatialAudioStreamHrtf;
	hr = spatialAudioClient->IsSpatialAudioStreamAvailable(__uuidof(ISpatialAudioObjectRenderStreamForHrtf), nullptr);
    if (FAILED(hr)) {
        std::cerr << "HRTF spatial audio not supported by spatial audio client." << std::endl;
        return 1;
    }

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

    SpatialAudioHrtfActivationParams streamParams;
    streamParams.ObjectFormat = &format;
    streamParams.StaticObjectTypeMask = AudioObjectType_None;
    streamParams.MinDynamicObjectCount = 0;
    streamParams.MaxDynamicObjectCount = min(maxDynamicObjectCount, 4);
    streamParams.Category = AudioCategory_GameEffects;
    streamParams.EventHandle = bufferCompletionEvent;
    streamParams.NotifyObject = nullptr;

	// What is specified by the Microsoft article, with some modifications
    SpatialAudioHrtfDistanceDecay decayModel;
    decayModel.CutoffDistance = 20.0f;
    decayModel.MaxGain = 3.98f;
    decayModel.MinGain = float(1.58439 * pow(10, -5));
    decayModel.Type = SpatialAudioHrtfDistanceDecayType::SpatialAudioHrtfDistanceDecay_CustomDecay;
    decayModel.UnityGainDistance = 1.0f;

    streamParams.DistanceDecay = &decayModel;

    SpatialAudioHrtfDirectivity directivity;
    directivity.Type = SpatialAudioHrtfDirectivityType::SpatialAudioHrtfDirectivity_Cardioid;
    directivity.Scaling = 1.0f;

    SpatialAudioHrtfDirectivityCardioid cardioid;
    cardioid.directivity = directivity;
	cardioid.Order = 0.0f; // Probably how forward focused the cardioid is, higher is narrower (probably useful for different frequencies)

    SpatialAudioHrtfDirectivityUnion directivityUnion;
    directivityUnion.Cardiod = cardioid; // Spelling typo
    streamParams.Directivity = &directivityUnion;

    SpatialAudioHrtfEnvironmentType environment = SpatialAudioHrtfEnvironmentType::SpatialAudioHrtfEnvironment_Small;
    streamParams.Environment = &environment;

    SpatialAudioHrtfOrientation orientation = { 1,0,0,0,1,0,0,0,1 }; // identity matrix
    streamParams.Orientation = &orientation;

    PROPVARIANT pv;
    PropVariantInit(&pv);
    pv.vt = VT_BLOB;
    pv.blob.cbSize = sizeof(streamParams);
    pv.blob.pBlobData = (BYTE*)&streamParams;

    hr = spatialAudioClient->ActivateSpatialAudioStream(&pv, __uuidof(spatialAudioStreamHrtf), (void**)&spatialAudioStreamHrtf);

    // Start streaming / rendering 
    hr = spatialAudioStreamHrtf->Start();

	std::vector<std::vector<float>> wavSamples = { leftWavSamples, rightWavSamples };
    std::vector<Windows::Foundation::Numerics::float3> positions = {
        Windows::Foundation::Numerics::float3(-1.0f, 0.0f, 0.0f),
        //Windows::Foundation::Numerics::float3(1.0f, 1.0f, 0.0f)
	};
    std::vector<Windows::Foundation::Numerics::float3> directions = {
		Windows::Foundation::Numerics::float3(1.0f, 0.0f, 0.0f), // Assumed unit vector at the moment
        Windows::Foundation::Numerics::float3(1.0f, 0.0f, 0.0f)
    };
	std::vector<size_t> channels = { 0, 1 };
    std::vector<Speaker3dObject> speakerObjects;

    Windows::Foundation::Numerics::float3 listenerDirection(0.0f, 0.0f, 1.0f);

    // Initializing sound object
    for (size_t i = 0; i < positions.size(); i++) {
        Microsoft::WRL::ComPtr<ISpatialAudioObjectForHrtf> audioObject;
        hr = spatialAudioStreamHrtf->ActivateSpatialAudioObjectForHrtf(AudioObjectType::AudioObjectType_Dynamic, &audioObject);

        DirectX::XMFLOAT4X4 rotationMatrix;
        DirectX::XMMATRIX rotation = CalculateEmitterOrientationMatrix(directions.at(i), listenerDirection);
        DirectX::XMStoreFloat4x4(&rotationMatrix, rotation); // Unload from vector registor
        SpatialAudioHrtfOrientation orientation = {
            rotationMatrix._11, rotationMatrix._12, rotationMatrix._13,
            rotationMatrix._21, rotationMatrix._22, rotationMatrix._23,
            rotationMatrix._31, rotationMatrix._32, rotationMatrix._33
        };

        speakerObjects.push_back({
            audioObject,
            positions.at(i),
			&orientation,
            20.0f,
            &wavSamples.at(channels.at(i)),
            static_cast<UINT>(wavSamples.at(channels.at(i)).size()),
            0
        });
    }

	std::cout << "Beginning spatial audio streaming." << std::endl;
    
	float rotateAngle = 0.0f;
    bool streaming = true;
    while (streaming) {
        // Wait for a signal from the audio-engine to start the next processing pass
        if (WaitForSingleObject(bufferCompletionEvent, 100) != WAIT_OBJECT_0) {
            break;
        }

        UINT32 availableDynamicObjectCount;
        UINT32 frameCount;

        hr = spatialAudioStreamHrtf->BeginUpdatingAudioObjects(&availableDynamicObjectCount, &frameCount);

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
                hr = it->audioObject->SetGain(it->gain); // Gain just doesn't work
				rotateAngle += 0.02f;
				it->position = Windows::Foundation::Numerics::float3(cos(rotateAngle), 0.0f, sin(rotateAngle));
				hr = it->audioObject->SetDistanceDecay(&decayModel); // Decay (not cutoff) depends on what direction the audio source is facing, some directions don't have decay at all
				hr = it->audioObject->SetDirectivity(&directivityUnion); // Directivity likely sets decay rate for different directions
                hr = it->audioObject->SetEnvironment(environment);
                hr = it->audioObject->SetOrientation(&orientation);

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

        hr = spatialAudioStreamHrtf->EndUpdatingAudioObjects();
    }

    // Stop the stream 
    hr = spatialAudioStreamHrtf->Stop();

    // We don't want to start again, so reset the stream to free it's resources.
    hr = spatialAudioStreamHrtf->Reset();

    CloseHandle(bufferCompletionEvent);

    return 0;
}