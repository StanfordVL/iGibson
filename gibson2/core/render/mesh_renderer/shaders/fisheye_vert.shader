#version 450
uniform mat4 V;
uniform mat4 P;
uniform mat4 pose_rot;
uniform mat4 pose_trans;
uniform vec3 instance_color;
uniform vec3 diffuse_color;

layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 texCoords;
out vec2 theCoords;
out vec3 Normal;
out vec3 FragPos;
out vec3 Normal_cam;
out vec3 Instance_color;
out vec3 Pos_cam;
out vec3 Diffuse_color;

vec2 pow_complex(vec2 a, float x) {
    float l = length(a);
    float angle = 0;
    if (l > 1e-8) {
        angle = atan(a.y, a.x);
    }
    l = pow(l,x);
    angle = angle * x;
    return vec2(l * cos(angle), l * sin(angle));
}

vec2 multiply_complex(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 divide_complex(vec2 a, vec2 b) {
    return multiply_complex(a, pow_complex(b, -1));
}

vec2[2] solve_2nd(vec2 a2, vec2 a1, vec2 a0){
    vec2 q = multiply_complex(a1, a1) - 4 * multiply_complex(a2, a0);
    vec2 x1 = divide_complex(-a1 + pow_complex(q, 0.5), 2 * a2);
    vec2 x2 = divide_complex(-a1 - pow_complex(q, 0.5), 2 * a2);
    return vec2[2](x1, x2);
};

vec2 solve_3rd(vec2 a3, vec2 a2, vec2 a1, vec2 a0) {
    vec2 D1 = -4.0*multiply_complex(pow_complex(a1,3),a3)
    + multiply_complex(pow_complex(a1,2),pow_complex(a2,2))
    - 4.0*multiply_complex(a0,pow_complex(a2,3))
    + 18.0*multiply_complex(multiply_complex(a0,a1),multiply_complex(a2,a3))
    - 27.0*multiply_complex(pow_complex(a0,2),pow_complex(a3,2));

    vec2 D2 = -2.0*pow_complex(a2,3) + 9.0*multiply_complex(multiply_complex(a1,a2),a3)
     - 27.0*multiply_complex(a0, pow_complex(a3,2));

    vec2 b2 = divide_complex(a2, a3);
    vec2 b1 = divide_complex(a1, a3);
    vec2 b0 = divide_complex(a0, a3);

    vec2 s0 = -b2;
    vec2 s1 = pow_complex((D2 + 3.0*pow_complex(-3.0*D1, 1.0/2.0))/2.0, 1.0/3.0);
    vec2 s2 = pow_complex((D2 - 3.0*pow_complex(-3.0*D1, 1.0/2.0))/2.0, 1.0/3.0);

    vec2 w = (-1+pow_complex(vec2(-3, 0), 1.0/2.0))/2.0;

    vec2 x0 = (s0 + s1 + s2)/3.0;
    //x1 = (s0 + w**2*s1 + w*s2)/3.0;
    //x2 = (s0 + w*s1 + w**2*s2)/3.0;

    return x0;
}

vec2[4] solve_4th(vec2 a4, vec2 a3, vec2 a2, vec2 a1, vec2 a0) {
    vec2 b3 = divide_complex(a3, a4);
    vec2 b2 = divide_complex(a2, a4);
    vec2 b1 = divide_complex(a1, a4);
    vec2 b0 = divide_complex(a0, a4);
    vec2 c3 = b3/4.0;
    vec2 p = b2 - 6.0*multiply_complex(c3, c3);
    vec2 q = b1 - 2.0*multiply_complex(b2, c3) + 8.0*pow_complex(c3, 3);
    vec2 r = b0 - multiply_complex(b1, c3) + multiply_complex(b2, multiply_complex(c3,c3))
     - 3.0*pow_complex(c3,4);

    vec2 u = solve_3rd(vec2(1,0), 2*p, pow_complex(p,2)-4.0*r, -pow_complex(q,2));
    vec2[2] res1 = solve_2nd(vec2(1,0), pow_complex(u, 1.0/2.0),
        (p+u)/2.0-divide_complex(multiply_complex(q, pow_complex(u, 1.0/2.0)/2.0),u));
    vec2 y1 = res1[0];
    vec2 y2 = res1[1];

    vec2[2] res2 = solve_2nd(vec2(1,0), -pow_complex(u, 1.0/2.0),
        (p+u)/2.0+divide_complex(multiply_complex(q, pow_complex(u, 1.0/2.0)/2.0),u));
    vec2 y3 = res2[0];
    vec2 y4 = res2[1];
    vec2 x1 = y1 - c3;
    vec2 x2 = y2 - c3;
    vec2 x3 = y3 - c3;
    vec2 x4 = y4 - c3;
    return vec2[4](x1, x2, x3, x4);
}

vec4 camera_model(vec3 point) {
    float cam_para1 = 6.156677914837426e-01;
    float cam_para2 = -8.608988802573333e-01;
    float cam_para3 = 8.521247230645207e-01;
    float cam_para4 = -7.732395849774419e-01;
    vec2 dis_center = vec2(3.119412049523769e+02, 3.194164203930683e+02) / 256.0 * FISHEYE_SIZE;
    float sig = length(point.xy);
    float A4 = 1.0;
    float A3 = -point.z/cam_para1;
    float A2 = cam_para2*sig*sig/cam_para1;
    float A1 = cam_para3*sig*sig*sig/cam_para1;
    float A0 = cam_para4*sig*sig*sig*sig/cam_para1;

    vec2[4] res4 = solve_4th(vec2(A4,0),vec2(A3,0),vec2(A2,0),vec2(A1,0),vec2(A0,0));
    float udd, vdd;
    float err = 100000000;
    float lambda;
    for (int i = 0; i < 4; i++) {
        lambda = res4[i].x;
        float u = point.x / lambda;
        float v = point.y / lambda;
        float ro = length(vec2(u,v));
        float w = cam_para1 + cam_para2*ro*ro + cam_para3*ro*ro*ro + cam_para4*ro*ro*ro*ro;
        if (err > abs(point.z - lambda * w)) {
            err = abs(point.z - lambda * w);
            udd = u * 1.2109375 * FISHEYE_SIZE + dis_center.x;
            vdd = v * 1.2109375 * FISHEYE_SIZE + dis_center.y;
        }
    };

    return vec4(udd,vdd,err,lambda);
}

void main() {

    gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);

    vec4 tmp_Position = V * pose_trans * pose_rot * vec4(position, 1);
    vec4 projection = camera_model(tmp_Position.xyz);

    float zfar = 100;
    float znear = 0.01;
    float l = length(tmp_Position.xyz);
    gl_Position.xyz = tmp_Position.xyz/l;
    gl_Position.z = gl_Position.z + 1;
    gl_Position.x = gl_Position.x / gl_Position.z;
    gl_Position.y = gl_Position.y / gl_Position.z;
    gl_Position.z = (l-znear) / (zfar - znear);
    gl_Position.w = 1;

    float angle;
    vec3 v1 = tmp_Position.xyz / length(tmp_Position.xyz);
    vec3 v2 = vec3(0,0,1);
    angle = acos(dot(v1, v2));
    if (angle > 2.5) gl_Position = vec4(0,0,100000,1);

    vec4 world_position4 = pose_trans * pose_rot * vec4(position, 1);
    FragPos = vec3(world_position4.xyz / world_position4.w); // in world coordinate
    Normal = normalize(mat3(pose_rot) * normal); // in world coordinate
    Normal_cam = normalize(mat3(V) * mat3(pose_rot) * normal); // in camera coordinate

    vec4 pos_cam4 = V * pose_trans * pose_rot * vec4(position, 1);
    Pos_cam = pos_cam4.xyz / pos_cam4.w;

    theCoords = texCoords;
    Instance_color = instance_color;
    Diffuse_color = diffuse_color;

}