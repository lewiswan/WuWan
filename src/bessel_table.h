// bessel_table.h
#pragma once
#include <array>
#include <boost/math/special_functions/bessel.hpp>

namespace BesselZeros {

// 编译时生成器模板
template <int N>
struct J0Table {
    static const std::array<double, N>& get() {
        static const std::array<double, N> table = []() {
            std::array<double, N> arr{};
            for (int i = 0; i < N; ++i) {
                arr[i] = boost::math::cyl_bessel_j_zero(0.0, i + 1);
            }
            return arr;
        }();
        return table;
    }
};

template <int N>
struct J1Table {
    static const std::array<double, N>& get() {
        static const std::array<double, N> arr = []() {
            std::array<double, N> arr{};
            for (int i = 0; i < N; ++i) {
                arr[i] = boost::math::cyl_bessel_j_zero(1.0, i + 1);
            }
            return arr;
        }();
        return arr;
    }
};

// 固定大小的表（程序启动时初始化一次）
inline const std::array<double, 120>& J0() {
    return J0Table<120>::get();
}

inline const std::array<double, 120>& J1() {
    return J1Table<120>::get();
}

} // namespace BesselZeros
