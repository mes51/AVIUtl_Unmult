#include "main.h"
#include "math.h"
#include <algorithm>
#include <omp.h>
#include <intrin.h>

#define SUPPORT_AVX (1 << 28)

inline __m128 horizontalMax(__m128 x)
{
    // SEE: https://stackoverflow.com/a/18616825
    __m128 max1 = _mm_permute_ps(x, _MM_PERM_AADC);
    __m128 max2 = _mm_max_ps(x, max1);
    __m128 max3 = _mm_permute_ps(max2, _MM_PERM_AAAB);
    __m128 max4 = _mm_max_ps(max2, max3);
    return _mm_permute_ps(max4, _MM_PERM_AAAA);
}

int unmultSimd(lua_State* L) {
    int* pixels = reinterpret_cast<int*>(lua_touserdata(L, 1));
    int w = static_cast<int>(lua_tointeger(L, 2));
    int h = static_cast<int>(lua_tointeger(L, 3));
    int length = w * h;

#pragma omp parallel for
    for (int i = 0; i < length; i++) {
        __m128i d = _mm_cvtsi32_si128(pixels[i]);
        __m128i ip = _mm_unpacklo_epi8(_mm_unpacklo_epi8(d, _mm_setzero_si128()), _mm_setzero_si128());
        __m128 p = _mm_cvtepi32_ps(ip);
        __m128 a = _mm_permute_ps(p, _MM_PERM_DDDD);

        if (_mm_comilt_ss(_mm_permute_ps(p, 255), _mm_set1_ps(255.0F)))
        {
            p = _mm_div_ps(_mm_mul_ps(p, a), _mm_set_ps(256.0F, 256.0F, 256.0F, 256.0F));
        }
        p = _mm_insert_ps(p, _mm_setzero_ps(), _MM_MK_INSERTPS_NDX(0, 3, 0));

        __m128 irate = horizontalMax(p);
        if (_mm_comigt_ss(irate, _mm_setzero_ps()))
        {
            __m128 t = _mm_mul_ps(p, _mm_div_ps(_mm_set_ps(255.0F, 255.0F, 255.0F, 255.0F), irate));
            __m128 ta = _mm_div_ps(_mm_mul_ps(p, _mm_set_ps(255.0F, 255.0F, 255.0F, 255.0F)), t);

            if (_mm_comigt_ss(_mm_permute_ps(ta, _MM_PERM_AAAA), _mm_setzero_ps()))
            {
                a = _mm_permute_ps(ta, _MM_PERM_AAAA);
            }
            else if (_mm_comigt_ss(_mm_permute_ps(ta, _MM_PERM_BBBB), _mm_setzero_ps()))
            {
                a = _mm_permute_ps(ta, _MM_PERM_BBBB);
            }
            else if (_mm_comigt_ss(_mm_permute_ps(ta, _MM_PERM_CCCC), _mm_setzero_ps()))
            {
                a = _mm_permute_ps(ta, _MM_PERM_CCCC);
            }
            else
            {
                pixels[i] = 0;
                continue;
            }

            p = _mm_insert_ps(t, a, _MM_MK_INSERTPS_NDX(0, 3, 0));
            ip = _mm_packus_epi16(_mm_packus_epi32(_mm_cvtps_epi32(p), _mm_setzero_si128()), _mm_setzero_si128());
            pixels[i] = _mm_cvtsi128_si32(ip);
        }
        else
        {
            pixels[i] = 0;
        }
    }

    return 0;
}

int unmult(lua_State* L) {
    Pixel* pixels = reinterpret_cast<Pixel*>(lua_touserdata(L, 1));
    int w = static_cast<int>(lua_tointeger(L, 2));
    int h = static_cast<int>(lua_tointeger(L, 3));
    int length = w * h;

#pragma omp parallel for
    for (int i = 0; i < length; i++) {
        float b = pixels[i].b;
        float g = pixels[i].g;
        float r = pixels[i].r;
        float a = pixels[i].a;

        if (a < 255.0F)
        {
            b = (b * a) / 256.0F;
            g = (g * a) / 256.0F;
            r = (r * a) / 256.0F;
        }

        float irate = std::max(std::max(r, g), b);
        if (irate > 0.0F)
        {
            float rate = 255.0F / irate;
            float tb = b * rate;
            float tg = g * rate;
            float tr = r * rate;

            if (tb > 0.0)
            {
                a = (unsigned char)std::min((b * 255.0F) / tb, 255.0F);
            }
            else if (tg > 0.0)
            {
                a = (unsigned char)std::min((g * 255.0F) / tg, 255.0F);
            }
            else if (tr > 0.0)
            {
                a = (unsigned char)std::min((r * 255.0F) / tr, 255.0F);
            }

            if (a > 0)
            {
                pixels[i].b = (unsigned char)std::min(roundf(tb), 255.0F);
                pixels[i].g = (unsigned char)std::min(roundf(tg), 255.0F);
                pixels[i].r = (unsigned char)std::min(roundf(tr), 255.0F);
                pixels[i].a = a;
            }
            else
            {
                pixels[i].b = 0;
                pixels[i].g = 0;
                pixels[i].r = 0;
                pixels[i].a = 0;
            }
        }
        else
        {
            pixels[i].b = 0;
            pixels[i].g = 0;
            pixels[i].r = 0;
            pixels[i].a = 0;
        }
    }

    return 0;
}

extern "C" {
    __declspec(dllexport) int luaopen_Unmult(lua_State* L) {
        static luaL_Reg functions[] = {
            { "unmult", nullptr },
            { nullptr, nullptr }
        };
        if (functions[0].func == nullptr)
        {
            int cpuId[4];
            __cpuid(cpuId, 1);

            if (cpuId[2] & SUPPORT_AVX)
            {
                functions[0].func = unmultSimd;
            }
            else
            {
                functions[0].func = unmult;
            }
        }

        luaL_register(L, "Unmult", functions);
        return 1;
    }
}
