// compile this file: c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cgl_utils.cpp ../glad/gl.cpp -o CGLUtils`python3-config --extension-suffix`
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glad/gl.h>
//#include <EGL/egl.h>
//#include <EGL/eglext.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <openvr.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

namespace py = pybind11;

// Class that handles the GLFW context
class GLFWRendererContext {
public:
	GLFWRendererContext(int w, int h) :renderHeight(h), renderWidth(w) {};

	GLFWwindow* window;
	int renderWidth;
	int renderHeight;
	int windowWidth;
	int windowHeight;

	struct VertexDataWindow {
		glm::vec2 position;
		glm::vec2 texCoord;

		//VertexDataWindow(const glm::vec2& pos, const glm::vec2& tex) : position(pos), texCoord(tex) {}
	};

	GLuint cwVAO, cwVBO, cwIndexBuffer, cwIndexSize;

	int init(bool shouldHideWindow) {
		// Initialize GLFW context and window
		if (!glfwInit()) {
			fprintf(stderr, "Failed to initialize GLFW.\n");
			exit(EXIT_FAILURE);
		}

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		// Hide GLFW window if user requests
		if (shouldHideWindow) {
			glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		}

		// Decrease size but retain aspect ratio
		windowWidth = renderWidth / 4;
		windowHeight = renderHeight / 4;

		window = glfwCreateWindow(windowWidth, windowHeight, "Gibson VR - Left Eye Output", NULL, NULL);
		if (window == NULL) {
			fprintf(stderr, "Failed to create GLFW window.\n");
			exit(EXIT_FAILURE);
		}
		glfwMakeContextCurrent(window);

		// Turns Vsync off (1 to turn it on)
		glfwSwapInterval(0);

		printf("Succesfully initialized both GLFW context and window!\n");

		return 0;
	}

	// Note: companion window only renders left eye so users can easily see what is going on
	void setupCompanionWindow() {
		std::vector<VertexDataWindow> windowVerts;

		VertexDataWindow w1, w2, w3, w4;
		w1.position = glm::vec2(-1, -1);
		w1.texCoord = glm::vec2(0, 0);

		w2.position = glm::vec2(-1, 1);
		w2.texCoord = glm::vec2(0, 1);

		w3.position = glm::vec2(1, 1);
		w3.texCoord = glm::vec2(1, 1);

		w4.position = glm::vec2(1, -1);
		w4.texCoord = glm::vec2(1, 0);

		// Left eye vertices and texture coordinates
		windowVerts.push_back(w1);
		windowVerts.push_back(w2);
		windowVerts.push_back(w3);
		windowVerts.push_back(w4);

		GLushort windowIndices[] = {
			0, 1, 2,
			0, 2, 3
		};

		cwIndexSize = _countof(windowIndices);

		glGenVertexArrays(1, &cwVAO);
		glGenBuffers(1, &cwVBO);

		glBindVertexArray(cwVAO);
		glBindBuffer(GL_ARRAY_BUFFER, cwVBO);
		glBufferData(GL_ARRAY_BUFFER, windowVerts.size() * sizeof(VertexDataWindow), &windowVerts[0], GL_STATIC_DRAW);

		glGenBuffers(1, &cwIndexBuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cwIndexBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, cwIndexSize * sizeof(GLushort), &windowIndices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexDataWindow), (void*)offsetof(VertexDataWindow, position));

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(VertexDataWindow), (void*)offsetof(VertexDataWindow, texCoord));

		glBindVertexArray(0);
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	void renderCompanionWindow(GLuint windowShaderProgram, GLuint leftEyeTexId) {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_DEPTH_TEST);
		glViewport(0, 0, windowWidth, windowHeight);
		glUseProgram(windowShaderProgram);
		glBindVertexArray(cwVAO);

		glBindTexture(GL_TEXTURE_2D, leftEyeTexId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glDrawElements(GL_TRIANGLES, cwIndexSize, GL_UNSIGNED_SHORT, 0);

		glBindVertexArray(0);
		glUseProgram(0);
		glfwSwapBuffers(window);
		// Return to render viewport
		glViewport(0, 0, renderWidth, renderHeight);
	}

	void flush_swap_glfw() {
		glFlush();
		glfwSwapBuffers(window);
	}

	void release() {
		glfwTerminate();
	}
};

// Class containing values and functions needed to run VR around Gibson renderer
// This class uses the OpenGL coordinate system, and converts to Gibson's ROS coordinate system upon returning matrices
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

	// Device data stored in VR coordinates
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

	glm::mat4 gibToVR;
	glm::mat4 vrToGib;

	// Initialize VRSystem class
	// TIMELINE: Call before any other method in this class
	VRSystem() :m_pHMD(NULL), renderWidth(0), renderHeight(0), nearClip(0.1f), farClip(30.0f) {};

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
		// Set gibToVR and vrToGib matrices
		setCoordinateTransformMatrices();

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
	// Note: GLM is column-major, whereas numpy is row major, so we need to tranpose view matrices before conversion
	// Note 2: Projection matrices are passed in to OpenGL assuming they are column-major, so we don't need to transpose them
	// TIMELINE: Call before rendering so the camera is set properly
	py::list preRenderVR() {
		py::array_t<float> leftEyeProjNp = py::array_t<float>({ 4,4 }, glm::value_ptr(leftEyeProj));
		py::array_t<float> rightEyeProjNp = py::array_t<float>({ 4,4 }, glm::value_ptr(rightEyeProj));

		glm::mat4 worldToHead = glm::inverse(hmdData.deviceTransform);

		leftEyeView = leftEyePos * worldToHead * gibToVR;
		rightEyeView = rightEyePos * worldToHead * gibToVR;

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
			// 0 is no error, 101 is no focus (happens at start of rendering)
			if (err != 0 && err != 101) {
				fprintf(stderr, "Compositor error: %d\n", err);
			}
		}
		else if (!strcmp(eye, "right")) {
			vr::Texture_t rightEyeTexture = { (void*)(uintptr_t)texID, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };;
			vr::EVRCompositorError err = vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
			if (err != 0 && err != 101) {
				fprintf(stderr, "Compositor error: %d\n", err);
			}
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

	// Returns device data in order: isValidData, position, rotation
	// Device type can be either hmd, left_controller or right_controller
	// Coordinates converted to Gibson ROS system from VR (OpenGL) coordinate system
	// TIMELINE: Call at any time after postRenderVR to poll the VR system for device data
	py::list getDataForVRDevice(char* deviceType) {
		bool isValid = false;

		py::array_t<float> positionData;
		py::array_t<float> rotationData;

		// TODO: Extend this to work with multiple headsets in future
		if (!strcmp(deviceType, "hmd")) {
			glm::vec3 transformedPos(vrToGib * glm::vec4(hmdData.devicePos, 1.0));
			positionData = py::array_t<float>({ 3, }, glm::value_ptr(transformedPos));
			rotationData = py::array_t<float>({ 4, }, glm::value_ptr(vrToGib * hmdData.deviceRot));
			isValid = hmdData.isValidData;
		}
		else if (!strcmp(deviceType, "left_controller")) {
			glm::vec3 transformedPos(vrToGib * glm::vec4(leftControllerData.devicePos, 1.0));
			positionData = py::array_t<float>({ 3, }, glm::value_ptr(transformedPos));
			rotationData = py::array_t<float>({ 4, }, glm::value_ptr(vrToGib * leftControllerData.deviceRot));
			isValid = leftControllerData.isValidData;
		}
		else if (!strcmp(deviceType, "right_controller")) {
			glm::vec3 transformedPos(vrToGib * glm::vec4(rightControllerData.devicePos, 1.0));
			positionData = py::array_t<float>({ 3, }, glm::value_ptr(transformedPos));
			rotationData = py::array_t<float>({ 4, }, glm::value_ptr(vrToGib * rightControllerData.deviceRot));
			isValid = rightControllerData.isValidData;
		}

		py::list deviceData;
		deviceData.append(isValid);
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

	// Sets coordinate transform matrices
	void setCoordinateTransformMatrices() {
		gibToVR[0] = glm::vec4(0.0, 0.0, 1.0, 0.0);
		gibToVR[1] = glm::vec4(1.0, 0.0, 0.0, 0.0);
		gibToVR[2] = glm::vec4(0.0, 1.0, 0.0, 0.0);
		gibToVR[3] = glm::vec4(0.0, 0.0, 0.0, 1.0);

		vrToGib[0] = glm::vec4(0.0, 1.0, 0.0, 0.0);
		vrToGib[1] = glm::vec4(0.0, 0.0, 1.0, 0.0);
		vrToGib[2] = glm::vec4(1.0, 0.0, 0.0, 0.0);
		vrToGib[3] = glm::vec4(0.0, 0.0, 0.0, 1.0);
	}
};

// CGLUtils functions
void render_meshrenderer_pre(bool msaa, GLuint fb1, GLuint fb2) {
    glBindFramebuffer(GL_FRAMEBUFFER, fb2);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (msaa) {
        glBindFramebuffer(GL_FRAMEBUFFER, fb1);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    glEnable(GL_DEPTH_TEST);
}

void render_meshrenderer_post() {
    glDisable(GL_DEPTH_TEST);
}


void glad_init() {
    /* if (!gladLoadGL(eglGetProcAddress)) {
        fprintf(stderr, "failed to load GL with glad.\n");
        exit(EXIT_FAILURE);
    } */

	if (!gladLoadGL(glfwGetProcAddress)) {
		fprintf(stderr, "Failed to load GL with glad.\n");
		exit(EXIT_FAILURE);
	}
}

std::string getstring_meshrenderer() {
    return reinterpret_cast<char const *>(glGetString(GL_VERSION));
}

void blit_buffer(int width, int height, GLuint fb1, GLuint fb2) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb1);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fb2);
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);

    for (int i = 0; i < 4; i++) {
        glReadBuffer(GL_COLOR_ATTACHMENT0+i);
		glDrawBuffer(GL_COLOR_ATTACHMENT0+i);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }
}

py::array_t<float> readbuffer_meshrenderer(char* mode, int width, int height, GLuint fb2) {
    glBindFramebuffer(GL_FRAMEBUFFER, fb2);
    if (!strcmp(mode, "rgb")) {
        glReadBuffer(GL_COLOR_ATTACHMENT0);
    }
    else if (!strcmp(mode, "normal")) {
        glReadBuffer(GL_COLOR_ATTACHMENT1);
    }
    else if (!strcmp(mode, "seg")) {
        glReadBuffer(GL_COLOR_ATTACHMENT2);
    }
    else if (!strcmp(mode, "3d")) {
        glReadBuffer(GL_COLOR_ATTACHMENT3);
    }
    else {
        fprintf(stderr, "unknown buffer mode.\n");
        exit(EXIT_FAILURE);
    }
    py::array_t<float> data = py::array_t<float>(4 * width * height);
    py::buffer_info buf = data.request();
    float* ptr = (float *) buf.ptr;
    glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, ptr);
    return data;
}

void clean_meshrenderer(std::vector<GLuint> texture1, std::vector<GLuint> texture2, std::vector<GLuint> fbo, std::vector<GLuint> vaos, std::vector<GLuint> vbos) {
    glDeleteTextures(texture1.size(), texture1.data());
    glDeleteTextures(texture2.size(), texture2.data());
    glDeleteFramebuffers(fbo.size(), fbo.data());
    glDeleteBuffers(vaos.size(), vaos.data());
    glDeleteBuffers(vbos.size(), vbos.data());
}

py::list setup_framebuffer_meshrenderer(int width, int height) {
    GLuint *fbo_ptr = (GLuint*)malloc(sizeof(GLuint));
    GLuint *texture_ptr = (GLuint*)malloc(5 * sizeof(GLuint));
    glGenFramebuffers(1, fbo_ptr);
    glGenTextures(5, texture_ptr);
    int fbo = fbo_ptr[0];
    int color_tex_rgb = texture_ptr[0];
    int color_tex_normal = texture_ptr[1];
    int color_tex_semantics = texture_ptr[2];
    int color_tex_3d = texture_ptr[3];
    int depth_tex = texture_ptr[4];

    glBindTexture(GL_TEXTURE_2D, color_tex_rgb);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	// Need these for compositor to work properly!
	// TODO: Make them a user-option or leave them in by default?
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glBindTexture(GL_TEXTURE_2D, color_tex_normal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, color_tex_semantics);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, color_tex_3d);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, depth_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, width, height, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex_rgb, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, color_tex_normal, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, color_tex_semantics, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, color_tex_3d, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0);
    glViewport(0, 0, width, height);
    GLenum *bufs = (GLenum*)malloc(4 * sizeof(GLenum));
    bufs[0] = GL_COLOR_ATTACHMENT0;
    bufs[1] = GL_COLOR_ATTACHMENT1;
    bufs[2] = GL_COLOR_ATTACHMENT2;
    bufs[3] = GL_COLOR_ATTACHMENT3;
    glDrawBuffers(4, bufs);
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    py::list result;
    result.append(fbo);
    result.append(color_tex_rgb);
    result.append(color_tex_normal);
    result.append(color_tex_semantics);
    result.append(color_tex_3d);
    result.append(depth_tex);

    return result;
}

py::list setup_framebuffer_meshrenderer_ms(int width, int height) {
    GLuint *fbo_ptr = (GLuint*)malloc(sizeof(GLuint));
    GLuint *texture_ptr = (GLuint*)malloc(5 * sizeof(GLuint));
    glGenFramebuffers(1, fbo_ptr);
    glGenTextures(5, texture_ptr);
    int fbo = fbo_ptr[0];
    int color_tex_rgb = texture_ptr[0];
    int color_tex_normal = texture_ptr[1];
    int color_tex_semantics = texture_ptr[2];
    int color_tex_3d = texture_ptr[3];
    int depth_tex = texture_ptr[4];
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_rgb);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA, width, height, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_normal);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA, width, height, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_semantics);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA, width, height, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color_tex_3d);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA, width, height, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, depth_tex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_DEPTH24_STENCIL8, width, height, GL_TRUE);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, color_tex_rgb, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D_MULTISAMPLE, color_tex_normal, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D_MULTISAMPLE, color_tex_semantics, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D_MULTISAMPLE, color_tex_3d, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D_MULTISAMPLE, depth_tex, 0);
    glViewport(0, 0, width, height);
    GLenum *bufs = (GLenum*)malloc(4 * sizeof(GLenum));
    bufs[0] = GL_COLOR_ATTACHMENT0;
    bufs[1] = GL_COLOR_ATTACHMENT1;
    bufs[2] = GL_COLOR_ATTACHMENT2;
    bufs[3] = GL_COLOR_ATTACHMENT3;
    glDrawBuffers(4, bufs);
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    py::list result;
    result.append(fbo);
    result.append(color_tex_rgb);
    result.append(color_tex_normal);
    result.append(color_tex_semantics);
    result.append(color_tex_3d);
    result.append(depth_tex);
    return result;
}

py::list compile_shader_meshrenderer(char* vertexShaderSource, char* fragmentShaderSource) {
    int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    int texUnitUniform = glGetUniformLocation(shaderProgram, "texUnit");
    py::list result;
    result.append(shaderProgram);
    result.append(texUnitUniform);
    return result;
}

py::list load_object_meshrenderer(int shaderProgram, py::array_t<float> vertexData) {
    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    GLuint VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    py::buffer_info buf = vertexData.request();
    float* ptr = (float *) buf.ptr;
    glBufferData(GL_ARRAY_BUFFER, vertexData.size()*sizeof(float), ptr, GL_STATIC_DRAW);
    GLuint positionAttrib = glGetAttribLocation(shaderProgram, "position");
    GLuint normalAttrib = glGetAttribLocation(shaderProgram, "normal");
    GLuint coordsAttrib = glGetAttribLocation(shaderProgram, "texCoords");
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(positionAttrib, 3, GL_FLOAT, GL_FALSE, 32, (void*)0);
    glVertexAttribPointer(normalAttrib, 3, GL_FLOAT, GL_FALSE, 32, (void*)12);
    glVertexAttribPointer(coordsAttrib, 2, GL_FLOAT, GL_TRUE, 32, (void*)24);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    py::list result;
    result.append(VAO);
    result.append(VBO);
    return result;
}

void render_softbody_instance(int vao, int vbo, py::array_t<float> vertexData) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    py::buffer_info buf = vertexData.request();
    float* ptr = (float *) buf.ptr;
    glBufferData(GL_ARRAY_BUFFER, vertexData.size()*sizeof(float), ptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void initvar_instance(int shaderProgram, py::array_t<float> V, py::array_t<float> P, py::array_t<float> pose_trans, py::array_t<float> pose_rot, py::array_t<float> lightpos, py::array_t<float> lightcolor) {
    glUseProgram(shaderProgram);
    float *Vptr = (float *) V.request().ptr;
    float *Pptr = (float *) P.request().ptr;
    float *transptr = (float *) pose_trans.request().ptr;
    float *rotptr = (float *) pose_rot.request().ptr;
    float *lightposptr = (float *) lightpos.request().ptr;
    float *lightcolorptr = (float *) lightcolor.request().ptr;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_trans"), 1, GL_FALSE, transptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_rot"), 1, GL_TRUE, rotptr);
    glUniform3f(glGetUniformLocation(shaderProgram, "light_position"), lightposptr[0], lightposptr[1], lightposptr[2]);
    glUniform3f(glGetUniformLocation(shaderProgram, "light_color"), lightcolorptr[0], lightcolorptr[1], lightcolorptr[2]);
}

void init_material_instance(int shaderProgram, float instance_color, py::array_t<float> diffuse_color, float use_texture) {
    float *diffuse_ptr = (float *) diffuse_color.request().ptr;
    glUniform3f(glGetUniformLocation(shaderProgram, "instance_color"), instance_color, 0, 0);
    glUniform3f(glGetUniformLocation(shaderProgram, "diffuse_color"), diffuse_ptr[0], diffuse_ptr[1], diffuse_ptr[2]);
    glUniform1f(glGetUniformLocation(shaderProgram, "use_texture"), use_texture);
}

void draw_elements_instance(bool flag, int texture_id, int texUnitUniform, int vao, int face_size, py::array_t<unsigned int> faces, GLuint fb) {
    glActiveTexture(GL_TEXTURE0);
    if (flag) glBindTexture(GL_TEXTURE_2D, texture_id);
    glUniform1i(texUnitUniform, 0);
    glBindVertexArray(vao);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    unsigned int *ptr = (unsigned int *) faces.request().ptr;
    glDrawElements(GL_TRIANGLES, face_size, GL_UNSIGNED_INT, ptr);
}

void initvar_instance_group(int shaderProgram, py::array_t<float> V, py::array_t<float> P, py::array_t<float> lightpos, py::array_t<float> lightcolor) {
    glUseProgram(shaderProgram);
    float *Vptr = (float *) V.request().ptr;
    float *Pptr = (float *) P.request().ptr;
    float *lightposptr = (float *) lightpos.request().ptr;
    float *lightcolorptr = (float *) lightcolor.request().ptr;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "V"), 1, GL_TRUE, Vptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "P"), 1, GL_FALSE, Pptr);
    glUniform3f(glGetUniformLocation(shaderProgram, "light_position"), lightposptr[0], lightposptr[1], lightposptr[2]);
    glUniform3f(glGetUniformLocation(shaderProgram, "light_color"), lightcolorptr[0], lightcolorptr[1], lightcolorptr[2]);
}

void init_material_pos_instance(int shaderProgram, py::array_t<float> pose_trans, py::array_t<float> pose_rot, float instance_color, py::array_t<float> diffuse_color, float use_texture) {
    float *transptr = (float *) pose_trans.request().ptr;
    float *rotptr = (float *) pose_rot.request().ptr;
    float *diffuse_ptr = (float *) diffuse_color.request().ptr;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_trans"), 1, GL_FALSE, transptr);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "pose_rot"), 1, GL_TRUE, rotptr);
    glUniform3f(glGetUniformLocation(shaderProgram, "instance_color"), instance_color, 0, 0);
    glUniform3f(glGetUniformLocation(shaderProgram, "diffuse_color"), diffuse_ptr[0], diffuse_ptr[1], diffuse_ptr[2]);
    glUniform1f(glGetUniformLocation(shaderProgram, "use_texture"), use_texture);
}


void render_tensor_pre(bool msaa, GLuint fb1, GLuint fb2) {

    glBindFramebuffer(GL_FRAMEBUFFER, fb2);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (msaa) {
        glBindFramebuffer(GL_FRAMEBUFFER, fb1);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    glEnable(GL_DEPTH_TEST);
}


void render_tensor_post() {
    glDisable(GL_DEPTH_TEST);
}

void cglBindVertexArray(int vao) {
    glBindVertexArray(vao);
}

void cglUseProgram(int shaderProgram) {
    glUseProgram(shaderProgram);
}

int loadTexture(std::string filename) {
	//    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
	//    w, h = img.size
	//
	//    img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

	//    img_data = np.frombuffer(img.tobytes(), np.uint8)
	//    #print(img_data.shape)
		//width, height = img.size
		// glTexImage2D expects the first element of the image data to be the
		// bottom-left corner of the image.  Subsequent elements go left to right,
		// with subsequent lines going from bottom to top.

		// However, the image data was created with PIL Image tostring and numpy's
		// fromstring, which means we have to do a bit of reorganization. The first
		// element in the data output by tostring() will be the top-left corner of
		// the image, with following values going left-to-right and lines going
		// top-to-bottom.  So, we need to flip the vertical coordinate (y).

	int w;
	int h;
	int comp;
	stbi_set_flip_vertically_on_load(true);
	unsigned char* image = stbi_load(filename.c_str(), &w, &h, &comp, STBI_rgb);

	if (image == nullptr)
		throw(std::string("Failed to load texture"));


	GLuint texture;
	glGenTextures(1, &texture);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB,
		GL_UNSIGNED_BYTE, image);
	glGenerateMipmap(GL_TEXTURE_2D);
	stbi_image_free(image);
	return texture;
}

int loadTextureWithAlpha(std::string filename) {
    int w;
    int h;
    int comp;
    stbi_set_flip_vertically_on_load(true);
	unsigned char* image = stbi_load(filename.c_str(), &w, &h, &comp, STBI_rgb_alpha);

    if(image == nullptr)
        throw(std::string("Failed to load texture"));


    GLuint texture;
    glGenTextures(1, &texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA,
                    GL_UNSIGNED_BYTE, image);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(image);
    return texture;
}

py::list setup_color_framebuffer(int w, int h) {
	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	py::list dataList;
	dataList.append(fbo);
	dataList.append(tex);

	return dataList;
}

void render_simple_color_to_fbo(GLuint fbo) {
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
}

PYBIND11_MODULE(CGLUtils, m) {
    m.doc() = "C++ OpenGL, GLFW and OpenVR bindings";

	// GLFW renderer context class
	py::class_<GLFWRendererContext>(m, "GLFWRendererContext")
		.def(py::init<int, int>())
		.def("init", &GLFWRendererContext::init)
		.def("setupCompanionWindow", &GLFWRendererContext::setupCompanionWindow)
		.def("renderCompanionWindow", &GLFWRendererContext::renderCompanionWindow)
		.def("flush_swap_glfw", &GLFWRendererContext::flush_swap_glfw)
		.def("release", &GLFWRendererContext::release);

	// VR system class
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

    // class MeshRenderer
    m.def("render_meshrenderer_pre", &render_meshrenderer_pre, "pre-executed functions in MeshRenderer.render");
    m.def("render_meshrenderer_post", &render_meshrenderer_post, "post-executed functions in MeshRenderer.render");
    m.def("getstring_meshrenderer", &getstring_meshrenderer, "return GL version string");
    m.def("readbuffer_meshrenderer", &readbuffer_meshrenderer, "read pixel buffer");
    m.def("glad_init", &glad_init, "init glad");
    m.def("clean_meshrenderer", &clean_meshrenderer, "clean meshrenderer");
    m.def("setup_framebuffer_meshrenderer", &setup_framebuffer_meshrenderer, "setup framebuffer in meshrenderer");
    m.def("setup_framebuffer_meshrenderer_ms", &setup_framebuffer_meshrenderer_ms, "setup framebuffer in meshrenderer with MSAA");
    m.def("blit_buffer", &blit_buffer, "blit buffer");

    m.def("compile_shader_meshrenderer", &compile_shader_meshrenderer, "compile vertex and fragment shader");
    m.def("load_object_meshrenderer", &load_object_meshrenderer, "load object into VAO and VBO");
    m.def("loadTexture", &loadTexture, "load texture function");
	m.def("loadTextureWithAlpha", &loadTextureWithAlpha, "load texture with alpha function");

    // class MeshRendererG2G
    m.def("render_tensor_pre", &render_tensor_pre, "pre-executed functions in MeshRendererG2G.render");
    m.def("render_tensor_post", &render_tensor_post, "post-executed functions in MeshRendererG2G.render");

    // class Instance
    m.def("render_softbody_instance", &render_softbody_instance, "render softbody in instance.render");
    m.def("initvar_instance", &initvar_instance, "init uniforms in instance.render");
    m.def("init_material_instance", &init_material_instance, "init materials in instance.render");
    m.def("draw_elements_instance", &draw_elements_instance, "draw elements in instance.render and instancegroup.render");

    // class InstanceGroup
    m.def("initvar_instance_group", &initvar_instance_group, "init uniforms in instancegroup.render");
    m.def("init_material_pos_instance", &init_material_pos_instance, "init materials and position in instancegroup.render");

    // misc
    m.def("cglBindVertexArray", &cglBindVertexArray, "binding function");
    m.def("cglUseProgram", &cglUseProgram, "binding function");

	// VR-related functions
	m.def("setup_color_framebuffer", &setup_color_framebuffer, "debugging function for setting up a simple color framebuffer");
	m.def("render_simple_color_to_fbo", &render_simple_color_to_fbo, "debugging function for rendering to a simple color framebuffer");
}