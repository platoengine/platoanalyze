//@HEADER
// ************************************************************************
//
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************
//@HEADER

// Must be included first on Intel-Phi systems due to
// redefinition of SEEK_SET in <mpi.h>.

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_Time.hpp>
#include <fstream>

#include "alg/ErrorHandling.hpp"
#include "alg/ParallelComm.hpp"
#include "alg/ParseInput.hpp"
#include "alg/Run.hpp"

void printTimingResults()
{
  const std::string tTimerFilter = ""; //"Analyze:"; // Only timers beginning with this string get summarized.
  const bool tAlwaysWriteLocal = false;
  const bool tWriteGlobalStats = true;
  const bool tWriteZeroTimers  = false;
  std::ofstream tTimingOutputFileStream ("plato_analyze_timing_summary.txt", std::ofstream::out);
  Teuchos::TimeMonitor::summarize(tTimingOutputFileStream, tAlwaysWriteLocal, tWriteGlobalStats, tWriteZeroTimers, 
                                  Teuchos::ECounterSetOp::Intersection, tTimerFilter);
  tTimingOutputFileStream.close();
}

int main(int aArgc, char** aArgv) {
  Plato::enable_floating_point_exceptions();

  Plato::Comm::Machine tMachine(&aArgc, &aArgv);

  Kokkos::initialize(aArgc, aArgv);

  Omega_h::Library tOmegaHLib(&aArgc, &aArgv);

  Teuchos::Time tTimeMng("Total Time", true);

  Teuchos::ParameterList tProblem =
      Plato::input_file_parsing(aArgc, aArgv, tMachine);

  bool tSuccess = true;
  int tReturnCode = EXIT_SUCCESS;

  try
  {
    Plato::run(&tOmegaHLib, tProblem, tMachine);
  }
  PLATO_CATCH_STATEMENTS(true, tSuccess);

  if (!tSuccess) tReturnCode = EXIT_FAILURE;

  printTimingResults();

  return tReturnCode;
}
