#version 450

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBitangent;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D albedoMap;
layout(set = 0, binding = 1) uniform sampler2D normalMap;
layout(set = 0, binding = 2) uniform sampler2D metallicRoughnessMap;
layout(set = 0, binding = 3) uniform sampler2D aoMap;

layout(push_constant) uniform PushConstants {
    vec3 cameraPos;
    vec3 lightPos;
    vec3 lightColor;
    float exposure;
} push;

const float PI = 3.14159265359;

// PBR functions
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    // Sample textures
    vec4 albedo = texture(albedoMap, inTexCoord);
    vec3 normal = normalize(2.0 * texture(normalMap, inTexCoord).rgb - 1.0);
    vec2 metallicRoughness = texture(metallicRoughnessMap, inTexCoord).bg;
    float ao = texture(aoMap, inTexCoord).r;
    
    float metallic = metallicRoughness.x;
    float roughness = metallicRoughness.y;
    
    // Calculate tangent space normal
    mat3 TBN = mat3(normalize(inTangent), normalize(inBitangent), normalize(inNormal));
    vec3 N = normalize(TBN * normal);
    vec3 V = normalize(push.cameraPos - inWorldPos);
    vec3 L = normalize(push.lightPos - inWorldPos);
    vec3 H = normalize(V + L);
    
    // Calculate reflectance at normal incidence
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo.rgb, metallic);
    
    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);   
    float G   = GeometrySmith(N, V, L, roughness);      
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
    
    vec3 numerator    = NDF * G * F; 
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;
    
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;
    
    float NdotL = max(dot(N, L), 0.0);        
    
    // Final color
    vec3 Lo = (kD * albedo.rgb / PI + specular) * push.lightColor * NdotL;
    vec3 ambient = vec3(0.03) * albedo.rgb * ao;
    
    vec3 color = ambient + Lo;
    
    // HDR tonemapping
    color = vec3(1.0) - exp(-color * push.exposure);
    
    // Gamma correction
    color = pow(color, vec3(1.0/2.2)); 
    
    outColor = vec4(color, albedo.a);
} 