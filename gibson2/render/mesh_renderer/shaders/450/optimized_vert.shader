#version 450

#define MAX_ARRAY_SIZE 1024

layout (std140) uniform TransformDataTrans {
    mat4 pose_trans_array[MAX_ARRAY_SIZE];
};

layout (std140) uniform TransformDataRot {
    mat4 pose_rot_array[MAX_ARRAY_SIZE];
};

layout (std140) uniform TransformDataLastTrans {
    mat4 pose_last_trans_array[MAX_ARRAY_SIZE];
};

layout (std140) uniform TransformDataLastRot {
    mat4 pose_last_rot_array[MAX_ARRAY_SIZE];
};

layout (std140) uniform Hidden {
    vec4 hidden_array[MAX_ARRAY_SIZE];
};

layout (std140) uniform UVData {
    vec4 uv_transform_param[MAX_ARRAY_SIZE];
};


in int gl_DrawID;

uniform mat4 V;
uniform mat4 last_V;
uniform mat4 lightV;
uniform mat4 P;
uniform mat4 lightP;

uniform vec3 instance_color;
uniform vec3 diffuse_color;
uniform int shadow_pass;

layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 texCoords;
layout (location=3) in vec3 tangent;
layout (location=4) in vec3 bitangent;

out vec2 theCoords;
out vec3 Normal_world;
out vec3 FragPos;
out vec3 Normal_cam;
out vec3 Instance_color;
out vec3 Pos_cam;
out vec3 Pos_cam_prev;
out vec3 Pos_cam_projected;
out vec3 Diffuse_color;
out mat3 TBN;
flat out int Draw_id;
out vec4 FragPosLightSpace;
out vec2 Optical_flow;

void main() {
    mat4 pose_trans = pose_trans_array[gl_DrawID];
    mat4 pose_rot = transpose(pose_rot_array[gl_DrawID]);
    mat4 last_trans = pose_last_trans_array[gl_DrawID];
    mat4 last_rot = transpose(pose_last_rot_array[gl_DrawID]);
    // Hidden is stored in the x element of the vector - y, z and w are currently unused and set to 1.0
    float hidden = hidden_array[gl_DrawID].x;
    vec4 world_position4;
    vec4 pos_cam4_projected;
    vec4 pos_cam4;
    vec4 Position_prev;
    vec4 pos_cam_prev4;

    if (shadow_pass == 1) {
        gl_Position = lightP * lightV * pose_trans * pose_rot * vec4(position, 1) * (1 - hidden);
        world_position4 = pose_trans * pose_rot * vec4(position, 1);
        FragPos = vec3(world_position4.xyz / world_position4.w);// in world coordinate
        Normal_world = normalize(mat3(pose_rot) * normal);// in world coordinate
        Normal_cam = normalize(mat3(lightV) * mat3(pose_rot) * normal);// in camera coordinate
        pos_cam4_projected = lightP * lightV * pose_trans * pose_rot * vec4(position, 1);
        pos_cam4 = lightV * pose_trans * pose_rot * vec4(position, 1);

        Optical_flow = vec2(0,0);
        Position_prev = vec4(0,0,0,0);
        pos_cam_prev4 = vec4(0,0,0,0);
        Pos_cam_prev = vec3(0,0,0);
    } else {
        gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1) * (1 - hidden);
        Position_prev = P * last_V * last_trans * last_rot * vec4(position, 1);
        Optical_flow = gl_Position.xy/gl_Position.w - Position_prev.xy/Position_prev.w;
        world_position4 = pose_trans * pose_rot * vec4(position, 1);
        FragPos = vec3(world_position4.xyz / world_position4.w);// in world coordinate
        Normal_world = normalize(mat3(pose_rot) * normal);// in world coordinate
        Normal_cam = normalize(mat3(V) * mat3(pose_rot) * normal);// in camera coordinate
        pos_cam4_projected = P * V * pose_trans * pose_rot * vec4(position, 1);
        pos_cam4 = V * pose_trans * pose_rot * vec4(position, 1);
        pos_cam_prev4 = last_V * last_trans * last_rot * vec4(position, 1);
        Pos_cam_prev = pos_cam_prev4.xyz / pos_cam_prev4.w;
    }

    Pos_cam = pos_cam4.xyz / pos_cam4.w;
    Pos_cam_projected = pos_cam4_projected.xyz / pos_cam4_projected.w;


    theCoords.x = (cos(uv_transform_param[gl_DrawID][2]) * texCoords.x * uv_transform_param[gl_DrawID][0])
                   - (sin(uv_transform_param[gl_DrawID][2]) * texCoords.y * uv_transform_param[gl_DrawID][1]);
    theCoords.y = (sin(uv_transform_param[gl_DrawID][2]) * texCoords.x * uv_transform_param[gl_DrawID][0])
                   + (cos(uv_transform_param[gl_DrawID][2]) * texCoords.y * uv_transform_param[gl_DrawID][1]);

    Instance_color = instance_color;
    Diffuse_color = diffuse_color;
    vec3 T = normalize(vec3(pose_trans * pose_rot * vec4(tangent,   0.0)));
    vec3 B = normalize(vec3(pose_trans * pose_rot * vec4(bitangent, 0.0)));
    vec3 N = normalize(vec3(pose_trans * pose_rot * vec4(normal,    0.0)));
    TBN = mat3(T, B, N);
    FragPosLightSpace = lightP * lightV * pose_trans * pose_rot * vec4(position, 1);
    Draw_id = gl_DrawID;
}