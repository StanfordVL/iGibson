#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/gl.h>
#include <openvr.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

namespace py = pybind11;

// Class containing values and functions needed to run VR around Gibson renderer
class VRSystem {
public:
    vr::IVRSystem* m_pHMD;
    uint32_t renderWidth;
	uint32_t renderHeight;
	float nearClip;
	float farClip;

	// Boolean indicating whether the user has request the VR camera to be at a specific location
	bool hasSetCamera;
	glm::vec3 currCameraPos;
    
	struct DeviceData {
		// standard 4x4 transform
		glm::mat4 deviceTransform;
		// x,y,z
		glm::vec3 devicePos;
		// w, x, y, z (quaternion)
		glm::vec4 deviceRot;
		// is device valid and being tracked
		bool isValidData = false;
		// index of current device in device array
		int index = -1;
	};
    
    DeviceData hmdData;
    DeviceData leftControllerData;
    DeviceData rightControllerData;
    
	// View matrices for both left and right eyes (only proj and view are actually returned to the user)
	glm::mat4 leftEyeProj;
	glm::mat4 leftEyePos;
	glm::mat4 leftEyeView;
	glm::mat4 rightEyeProj;
	glm::mat4 rightEyePos;
	glm::mat4 rightEyeView;

	// Initialize VRSystem class
	// TIMELINE: Call before any other method in this class
	VRSystem():m_pHMD(NULL), renderWidth(0), renderHeight(0), nearClip(0.1f), farClip(30.0f) {};
    
    // Initialize the VR system and compositor and return recommended dimensions
    // TIMELINE: Call during init of renderer, before height/width are set
    py::list initVR() {
        // Initialize VR systems
        if (!vr::VR_IsRuntimeInstalled()) {
            fprintf(stderr, "VR runtime not installed.\n");
            exit(EXIT_FAILURE);
        }
        
        vr::EVRInitError eError = vr::VRInitError_None;
        m_pHMD = vr::VR_Init(&eError, vr::VRApplication_Scene);
        
        if (eError != vr::VRInitError_None) {
			fprintf(stderr, "Unable to initialize VR runtime.\n");
            exit(EXIT_FAILURE);
        }
        
        if (!vr::VRCompositor()) {
            fprintf(stderr, "Unable to intialize VR compositor.\n");
        }

		leftEyeProj = getHMDEyeProjection(vr::Eye_Left);
		leftEyePos = getHMDEyePose(vr::Eye_Left);
		rightEyeProj = getHMDEyeProjection(vr::Eye_Right);
		rightEyePos = getHMDEyePose(vr::Eye_Right);
        
		m_pHMD->GetRecommendedRenderTargetSize(&renderWidth, &renderHeight);
        
        py::list renderDims;
        renderDims.append((int)renderWidth);
        renderDims.append((int)renderHeight);

		hasSetCamera = false;
        
        return renderDims;
    }

	// Sets the position of the VR camera
	// TIMELINE: Call before preRenderVR - takes one frame to update camera position
	void setVRCamera(float x, float y, float z) {
		currCameraPos = glm::vec3(x, y, z);
		hasSetCamera = true;
	}

	// Resets the VR camera to use the position of the HMD as its location
	// TIMELINE: Call before preRenderVR - takes one frame to update camera position
	void resetVRCamera() {
		hasSetCamera = false;
	}

	// Returns the projection and view matrices for the left and right eyes, to be used in rendering
	// Returns in order Left P, left V, right P, right V
	// Note: GLM is column-major, whereas numpy is row major, so we need to tranpose before conversion
	// TIMELINE: Call before rendering so the camera is set properly
	py::list preRenderVR() {
		py::array_t<float> leftEyeProjNp = py::array_t<float>({ 4,4 }, glm::value_ptr(glm::transpose(leftEyeProj)));
		py::array_t<float> rightEyeProjNp = py::array_t<float>({ 4,4 }, glm::value_ptr(glm::transpose(rightEyeProj)));

		glm::mat4 worldToHead = glm::inverse(hmdData.deviceTransform);

		leftEyeView = leftEyePos * worldToHead;
		rightEyeView = rightEyePos * worldToHead;

		py::array_t<float> leftEyeViewNp = py::array_t<float>({ 4,4 }, glm::value_ptr(glm::transpose(leftEyeView)));
		py::array_t<float> rightEyeViewNp = py::array_t<float>({ 4,4 }, glm::value_ptr(glm::transpose(rightEyeView)));

		py::list eyeMats;
		eyeMats.append(leftEyeProjNp);
		eyeMats.append(leftEyeViewNp);
		eyeMats.append(rightEyeProjNp);
		eyeMats.append(rightEyeViewNp);

		return eyeMats;
	}
    
    // Called after the renderer has finished rendering a single eye
    // TIMELINE: Call immediately after rendering for current eye is finished
	void postRenderVRForEye(char* eye, GLuint texID) {
		if (!strcmp(eye, "left")) {
			vr::Texture_t leftEyeTexture = { (void*)(uintptr_t)texID, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
			vr::EVRCompositorError err = vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
		}
		else if (!strcmp(eye, "right")) {
			vr::Texture_t rightEyeTexture = { (void*)(uintptr_t)texID, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };;
			vr::EVRCompositorError err = vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
		}
	}
	
	// Called after both eyes have been rendered 
	// Tell the compositor to begin work immediately instead of waiting for the next WaitGetPoses() call if the user wants
	// And then update VR data
	// TIMELINE: Call immediately after calling postRenderVRForEye on both left and right eyes
	void postRenderVRUpdate(bool shouldHandoff) {
		if (shouldHandoff) {
			vr::VRCompositor()->PostPresentHandoff();
		}

		updateVRData();
	}
    
    // Returns device data in order: isValidData, transform, position, rotation
    // Device type can be either hmd, left_controller or right_controller
    // TIMELINE: Call at any time after postRenderVR to poll the VR system for device data
    py::list getDataForVRDevice(char* deviceType) {
		bool isValid = false;

        py::array_t<float> transformData;
        py::array_t<float> positionData;
        py::array_t<float> rotationData;
        
        if (!strcmp(deviceType, "hmd")) {
			transformData = py::array_t<float>({ 4,4 }, glm::value_ptr(hmdData.deviceTransform));
			positionData = py::array_t<float>({ 3, }, glm::value_ptr(hmdData.devicePos));
			rotationData = py::array_t<float>({ 4, }, glm::value_ptr(hmdData.deviceRot));
            isValid = hmdData.isValidData;
        }
        else if (!strcmp(deviceType, "left_controller")) {
			transformData = py::array_t<float>({ 4,4 }, glm::value_ptr(leftControllerData.deviceTransform));
			positionData = py::array_t<float>({ 3, }, glm::value_ptr(leftControllerData.devicePos));
			rotationData = py::array_t<float>({ 4, }, glm::value_ptr(leftControllerData.deviceRot));
            isValid = leftControllerData.isValidData;
        }
        else if (!strcmp(deviceType, "right_controller")) {
			transformData = py::array_t<float>({ 4,4 }, glm::value_ptr(rightControllerData.deviceTransform));
			positionData = py::array_t<float>({ 3, }, glm::value_ptr(rightControllerData.devicePos));
			rotationData = py::array_t<float>({ 4, }, glm::value_ptr(rightControllerData.deviceRot));
            isValid = rightControllerData.isValidData;
        }
        
        py::list deviceData;
        deviceData.append(isValid);
		deviceData.append(transformData);
        deviceData.append(positionData);
        deviceData.append(rotationData);
        
        return deviceData;
    }

    // Polls for VR events, such as button presses
    // TODO: Implement this for button/trigger interactivity!
    void pollVREvents() {
		printf("VR event system not implemented yet!");
	}
    
    // Releases and cleans up VR system
    // TIMELINE: Call when the renderer shuts down
    void releaseVR() {
        vr::VR_Shutdown();
        m_pHMD = NULL;
    }
    
private:
	// Calls WaitGetPoses and updates all hmd and controller transformations
	void updateVRData() {
		hmdData.isValidData = false;
		leftControllerData.isValidData = false;
		rightControllerData.isValidData = false;

		vr::TrackedDevicePose_t trackedDevices[vr::k_unMaxTrackedDeviceCount];
		vr::VRCompositor()->WaitGetPoses(trackedDevices, vr::k_unMaxTrackedDeviceCount, NULL, 0);

		for (unsigned int idx = 0; idx < vr::k_unMaxTrackedDeviceCount; idx++) {
			if (!trackedDevices[idx].bPoseIsValid || !m_pHMD->IsTrackedDeviceConnected(idx)) continue;

			vr::HmdMatrix34_t transformMat = trackedDevices[idx].mDeviceToAbsoluteTracking;
			vr::ETrackedDeviceClass trackedDeviceClass = m_pHMD->GetTrackedDeviceClass(idx);

			if (trackedDeviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_HMD) {
				hmdData.index = idx;
				hmdData.isValidData = true;
				if (hasSetCamera) {
					SetSteamVRMatrixPos(currCameraPos, transformMat);
				}
				hmdData.deviceTransform = convertSteamVRMatrixToGlmMat4(transformMat);
				hmdData.devicePos = getPositionFromSteamVRMatrix(transformMat);
				hmdData.deviceRot = getRotationFromSteamVRMatrix(transformMat);
			}
			else if (trackedDeviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_Controller) {
				vr::ETrackedControllerRole role = m_pHMD->GetControllerRoleForTrackedDeviceIndex(idx);
				if (role == vr::TrackedControllerRole_Invalid) {
					continue;
				}
				else if (role == vr::TrackedControllerRole_LeftHand) {
					leftControllerData.index = idx;
					leftControllerData.isValidData = true;
					leftControllerData.deviceTransform = convertSteamVRMatrixToGlmMat4(transformMat);
					leftControllerData.devicePos = getPositionFromSteamVRMatrix(transformMat);
					leftControllerData.deviceRot = getRotationFromSteamVRMatrix(transformMat);
				}
				else if (role == vr::TrackedControllerRole_RightHand) {
					rightControllerData.index = idx;
					rightControllerData.isValidData = true;
					rightControllerData.deviceTransform = convertSteamVRMatrixToGlmMat4(transformMat);
					rightControllerData.devicePos = getPositionFromSteamVRMatrix(transformMat);
					rightControllerData.deviceRot = getRotationFromSteamVRMatrix(transformMat);
				}
			}
		}
	}

	// Sets the position in a SteamVR Matrix
	void SetSteamVRMatrixPos(glm::vec3& pos, vr::HmdMatrix34_t& mat) {
		mat.m[0][3] = pos[0];
		mat.m[1][3] = pos[1];
		mat.m[2][3] = pos[2];
	}

	// Converts a SteamVR Matrix to a glm mat4
	glm::mat4 convertSteamVRMatrixToGlmMat4(const vr::HmdMatrix34_t& matPose) {
		glm::mat4 mat(
			matPose.m[0][0], matPose.m[1][0], matPose.m[2][0], 0.0,
			matPose.m[0][1], matPose.m[1][1], matPose.m[2][1], 0.0,
			matPose.m[0][2], matPose.m[1][2], matPose.m[2][2], 0.0,
			matPose.m[0][3], matPose.m[1][3], matPose.m[2][3], 1.0f
		);
		return mat;
	}

	// Gets position of HMD
	glm::vec3 getPositionFromSteamVRMatrix(vr::HmdMatrix34_t& matrix) {
		return glm::vec3(matrix.m[0][3], matrix.m[1][3], matrix.m[2][3]);
	}

	// Gets rotation of HMD in vec4 form
	glm::vec4 getRotationFromSteamVRMatrix(vr::HmdMatrix34_t& matrix) {
		glm::vec4 q;

		q[0] = (float)sqrt(fmax(0, 1 + matrix.m[0][0] + matrix.m[1][1] + matrix.m[2][2])) / 2;
		q[1] = (float)sqrt(fmax(0, 1 + matrix.m[0][0] - matrix.m[1][1] - matrix.m[2][2])) / 2;
		q[2] = (float)sqrt(fmax(0, 1 - matrix.m[0][0] + matrix.m[1][1] - matrix.m[2][2])) / 2;
		q[3] = (float)sqrt(fmax(0, 1 - matrix.m[0][0] - matrix.m[1][1] + matrix.m[2][2])) / 2;
		q[1] = copysign(q[1], matrix.m[2][1] - matrix.m[1][2]);
		q[2] = copysign(q[2], matrix.m[0][2] - matrix.m[2][0]);
		q[3] = copysign(q[2], matrix.m[1][0] - matrix.m[0][1]);

		return q;
	}

	// Generates a projection matrix for the specified eye (left or right)
	glm::mat4 getHMDEyeProjection(vr::Hmd_Eye eye) {
		vr::HmdMatrix44_t mat = m_pHMD->GetProjectionMatrix(eye, nearClip, farClip);

		glm::mat4 eyeProjMat(
			mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
			mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
			mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
			mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]
		);

		return eyeProjMat;
	}

	// Generates a pose matrix for the specified eye (left or right)
	glm::mat4 getHMDEyePose(vr::Hmd_Eye eye) {
		vr::HmdMatrix34_t mat = m_pHMD->GetEyeToHeadTransform(eye);

		glm::mat4 eyeToHead(
			mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0,
			mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0,
			mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0,
			mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0f
		);

		// Return the head to eye transform
		return glm::inverse(eyeToHead);
	}

	// Print string version of mat4 for debugging purposes
	void printMat4(glm::mat4& m) {
		printf(glm::to_string(m).c_str());
		printf("\n");
	}
};

PYBIND11_MODULE(VRUtils, m) {
    m.doc() = "VRUtilities module using C++ OpenVR bindings";
    
	py::class_<VRSystem>(m, "VRSystem")
		.def(py::init())
		.def("initVR", &VRSystem::initVR)
		.def("setVRCamera", &VRSystem::setVRCamera)
		.def("resetVRCamera", &VRSystem::resetVRCamera)
		.def("preRenderVR", &VRSystem::preRenderVR)
		.def("postRenderVRForEye", &VRSystem::postRenderVRForEye)
		.def("postRenderVRUpdate", &VRSystem::postRenderVRUpdate)
		.def("getDataForVRDevice", &VRSystem::getDataForVRDevice)
        .def("pollVREvents", &VRSystem::pollVREvents)
        .def("releaseVR", &VRSystem::releaseVR);
    
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
