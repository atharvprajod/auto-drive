#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inTangent;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outTexCoord;
layout(location = 3) out vec3 outTangent;
layout(location = 4) out vec3 outBitangent;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 proj;
} push;

void main() {
    vec4 worldPos = push.model * vec4(inPosition, 1.0);
    gl_Position = push.proj * push.view * worldPos;
    
    outWorldPos = worldPos.xyz;
    outNormal = mat3(push.model) * inNormal;
    outTexCoord = inTexCoord;
    outTangent = mat3(push.model) * inTangent;
    outBitangent = cross(outNormal, outTangent);
} 