#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

#include "dlpack/dlpack.h"
#include "tvm/driver/driver_api.h"
#include "tvm/runtime/container/array.h"
#include "tvm/runtime/data_type.h"
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/logging.h"
#include "tvm/target/target.h"
#include <tvm/te/tensor.h>
#include <tvm/te/schedule.h>
#include <tvm/te/operation.h>

using namespace std;
using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::te;

class TVMFilterBenchFixture : public benchmark::Fixture
{
 public:
  void SetUp(benchmark::State& state)
  {
    auto n = Var("n");
    Array<PrimExpr> shape {n};
    static const std::string targetStr{"llvm -mcpu=skylake-avx512"};
    size_t bitsUsed = 32;

    auto emptyVar = Var("emptyVar", DataType::Int(bitsUsed));
    auto firstFilterVar = Var("firstFilterVar", DataType::Int(bitsUsed));
    auto secFilterVar = Var("secFilterVar", DataType::Int(bitsUsed));
    auto src = placeholder(shape, DataType::Int(bitsUsed), "emptyVar");
    Tensor firstFilterOut = compute(src->shape, [&src, &emptyVar, &firstFilterVar](tvm::PrimExpr i) {
      return if_then_else(src[i] == firstFilterVar, src[i], emptyVar);
    });
    Tensor secFilterOut = compute(src->shape, [&src, &emptyVar, &firstFilterOut, &secFilterVar](tvm::PrimExpr i) {
      return if_then_else(firstFilterOut[i] == secFilterVar, firstFilterOut[i], emptyVar);
    });
    Tensor ridsOut = compute(src->shape, [&src, &secFilterOut, &emptyVar](tvm::PrimExpr i) {
      return if_then_else(secFilterOut[i] == emptyVar, i, 8192);
    });

    // set schedule
    Schedule s = create_schedule({firstFilterOut->op, secFilterOut->op, ridsOut->op});

    // build a module
    std::unordered_map<Tensor, Buffer> binds;
    auto args = Array<ObjectRef>({src, firstFilterOut, secFilterOut, ridsOut, emptyVar,
    firstFilterVar, secFilterVar});
    auto lowered = LowerSchedule(s, args, "int642filters", binds);
    // cerr << lowered << endl;

    auto target = Target(targetStr);
    auto targetHost = Target(targetStr);
    vecAddMod = build(lowered, target, targetHost);
    vecAddFunc = vecAddMod->GetFunction("int642filters");
    // Lanes might affect SIMD used
    int ndim = 1;
    int dtype_code = kDLInt;
    int dtype_bits = bitsUsed;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;
    int64_t shapeArr[1] = {blockSize};

    // C API funcs TBR
    TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &srcTensor);
    TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &firstFilterOutTensor);
    TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &secFilterOutTensor);
    TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &ridsOutTensor);

    for (int i = 0; i < shapeArr[0]; ++i)
    {
        static_cast<int32_t*>(srcTensor->data)[i] = i;
    }
  }

  // to avoid gcc compile time warning
  void SetUp(const benchmark::State& state)
  {
    SetUp(const_cast<benchmark::State&>(state));
  }

  constexpr const static size_t blockSize=8192;
  PackedFunc vecAddFunc;
  Module vecAddMod;
  DLTensor* srcTensor;
  DLTensor* firstFilterOutTensor;
  DLTensor* secFilterOutTensor;
  DLTensor* ridsOutTensor;
};

BENCHMARK_DEFINE_F(TVMFilterBenchFixture, TVM2filtersInt64)(benchmark::State& state)
{
  for (auto _ : state)
  {



// n = te.var("n")
// empty_var = te.var("empty_var", dtype='int64')
// first_filter_var = te.var("first_filter_var", dtype='int64')
// sec_filter_var = te.var("sec_filter_var", dtype='int64')
// SRC = te.placeholder((n,), dtype='int64', name="SRC")
// FIRST_FILTER_OUT = te.compute(SRC.shape,
//                               lambda i: te.if_then_else(
//                                   SRC[i] == first_filter_var, SRC[i], empty_var),
//                               name="FIRST_FILTER_OUT",
//                               )
// SEC_FILTER_OUT = te.compute(SRC.shape,
//                             lambda i: te.if_then_else(
//                                 FIRST_FILTER_OUT[i] == sec_filter_var, FIRST_FILTER_OUT[i], empty_var),
//                             name="SEC_FILTER_OUT",
//                             )
// #SEC_FILTER_OUT_SORTED = topi.sort(SEC_FILTER_OUT)

// RID_OUT = te.compute(SRC.shape,
//                      lambda i: te.if_then_else(
//                          SEC_FILTER_OUT[i] == empty_var, i, BLOCK_SIZE),
//                      name="RID_OUT"
//                      )

    // define algorithm

    // auto n = Var("n");
    // Array<PrimExpr> shape {n};
    // auto emptyVar = Var("emptyVar");
    // auto firstFilterVar = Var("firstFilterVar");
    // auto secFilterVar = Var("secFilterVar");
    // auto src = placeholder(shape, tvm::DataType::Int(64), "emptyVar");
    // Tensor firstFilterOut = compute(src->shape, [&src, &emptyVar, &firstFilterVar](tvm::PrimExpr i) {
    //   return if_then_else(src[i] == firstFilterVar, src[i], emptyVar);
    // });
    // Tensor secFilterOut = compute(src->shape, [&src, &emptyVar, &firstFilterOut, &secFilterVar](tvm::PrimExpr i) {
    //   return if_then_else(firstFilterOut[i] == secFilterVar, firstFilterOut[i], emptyVar);
    // });
    // Tensor ridsOut = compute(src->shape, [&src, &secFilterOut, &emptyVar](tvm::PrimExpr i) {
    //   return if_then_else(secFilterOut[i] == emptyVar, i, 8192);
    // });

    // // set schedule
    // Schedule s = create_schedule({firstFilterOut->op, secFilterOut->op, ridsOut->op});

    // // build a module
    // std::unordered_map<Tensor, Buffer> binds;
    // auto args = Array<ObjectRef>({src, firstFilterOut, secFilterOut, ridsOut, emptyVar,
    // firstFilterVar, secFilterVar});
    // auto lowered = LowerSchedule(s, args, "int642filters", binds);
    // // cerr << lowered << endl;

    // auto target = Target(targetStr);
    // auto targetHost = Target(targetStr);
    // Module mod = build(lowered, target, targetHost);
    // PackedFunc vecAddFunc = mod->GetFunction("int642filters");


    // cout << vecAddMod->GetSource() << endl;
    // state.PauseTiming();

    // DLTensor* a;
    // DLTensor* b;
    // DLTensor* c;

    // auto target = Target(targetStr);
    // auto targetHost = Target(targetStr);
    // vecAddMod = build(lowered, target, targetHost);
    // vecAddFunc = vecAddMod->GetFunction("int642filters");
    // int ndim = 1;
    // int dtype_code = kDLInt;
    // int dtype_bits = 64;
    // int dtype_lanes = 1;
    // int device_type = kDLCPU;
    // int device_id = 0;
    // int64_t shapeArr[1] = {8192};

    // TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
    //                 device_type, device_id, &a);
    // TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
    //                 device_type, device_id, &b);
    // TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
    //                 device_type, device_id, &c);
    // for (int i = 0; i < shapeArr[0]; ++i) {
    //     static_cast<int64_t*>(a->data)[i] = i;
    //     static_cast<int64_t*>(b->data)[i] = i*10;
    // }

    // state.ResumeTiming();
    TVMValue emptyVar;
    emptyVar.v_int64 = 0xFFFFFFFE;
    TVMValue firstFilterVar;
    firstFilterVar.v_int64 = 20;
    TVMValue secFilterVar;
    secFilterVar.v_int64 = 20000;
    TVMArgValue emptyVarArg(emptyVar, kTVMArgInt);
    TVMArgValue firstFilterVarArg{firstFilterVar, kTVMArgInt};
    TVMArgValue secFilterVarArg{secFilterVar, kTVMArgInt};

    // Call
    for (size_t i = 0; i < 1000000; i += blockSize)
    {
        vecAddFunc(srcTensor, firstFilterOutTensor, secFilterOutTensor,
          ridsOutTensor, emptyVarArg, firstFilterVarArg, secFilterVarArg);
    }
  }
}

BENCHMARK_REGISTER_F(TVMFilterBenchFixture, TVM2filtersInt64);
BENCHMARK_MAIN();