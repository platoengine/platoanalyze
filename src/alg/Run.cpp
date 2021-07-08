//@HEADER
// ************************************************************************
// Copyright 2018 National Technology & Engineering Solutions of Sandia,
// LLC (NTESS).  Under the terms of Contract DE-NA0003525, the U.S. 
// Government retains certain rights // in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions 
// are met:
//
// 1. Redistributions of source code must retain the above copyright 
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright 
//    notice, this list of conditions and the following disclaimer in 
//    the documentation and/or other materials provided with the 
//    distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER 
// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are 
// those of the authors and should not be interpreted as representing 
// official policies, either expressed or implied, of NTESS or the U.S. 
// Government.
// ************************************************************************
//@HEADER

#include <cstdlib>

#include <Teuchos_ParameterList.hpp>
#include <Kokkos_Core.hpp>

#include "AnalyzeConfig.hpp"
#include "alg/Run.hpp"

#include "alg/ErrorHandling.hpp"
#include "PlatoDriver.hpp"

namespace Plato
{

#if defined(KOKKOS_HAVE_CUDA)
void run_cuda_query(Comm::Machine machine)
{
    const size_t comm_rank = Comm::rank(machine);
    std::cout << "P" << comm_rank
    << ": Cuda device_count = " << Kokkos::Cuda::detect_device_count()
    << std::endl;
}
#endif

/******************************************************************************//**
 * \brief Manages Check for input mesh file errors.
 * \param [in] aProblem input file parameter list
 **********************************************************************************/
void error_check(Teuchos::ParameterList& aProblem)
{
    if(aProblem.isParameter("Input Mesh") == false)
    {
        std::stringstream tMsg;
        tMsg << "Input Mesh Error: 'Input Mesh' Parameter Keyword is NOT defined.";
        THROWERR(tMsg.str().c_str())
    }
    auto tInputMesh = aProblem.get<std::string>("Input Mesh");

    auto tPosition = tInputMesh.find(".exo");
    if(tPosition == std::string::npos)
    {
        std::stringstream tMsg;
        tMsg << "Input Mesh Error: 'Input Mesh' file extension was not provided. "  <<
                "'Input Mesh' argument is set to '" << tInputMesh << "'. File extension .exo is missing. " <<
                "'Input Mesh' argument should be defined as follows: 'filename.exo'.";
        THROWERR(tMsg.str().c_str())
    }

    auto tPhysicsString = aProblem.get<std::string>("Physics", "Plato Driver");
    if(tPhysicsString != "Plato Driver")
    {
        std::stringstream tMsg;
        tMsg << "Input File Error: 'Physics' Parameter Keyword is NOT defined. "
            << "The 'Physics' Parameter Keyword is set to " << tPhysicsString.c_str()
            << ". The 'Physics' Parameter Keyword should be set to 'Plato Driver'.";
        THROWERR(tMsg.str().c_str())
    }
}
// function error_check

/******************************************************************************//**
 * \brief Run a Plato Analyze problem
 * \param [in] aOmegaHLib pointer to Omega_h library
 * \param [in] aProblem   input file parameter list
 * \param [in] aMachine   MPI wrapper
 **********************************************************************************/
void run(
    Omega_h::Library*       aOmegaHLib,
    Teuchos::ParameterList& aProblem,
    Comm::Machine           aMachine)
{
    if(Comm::rank(aMachine) == 0)
    {

        std::cout << "\nRunning Plato Analyze version " << version_major << "."
                  << version_minor << "." << version_patch << std::endl;
    }

    if(aProblem.get<bool>("Query"))
    {
        if(Comm::rank(aMachine) == 0)
        {
            const unsigned tNumaCount = Kokkos::hwloc::get_available_numa_count();
            const unsigned tCoresPerNuma = Kokkos::hwloc::get_available_cores_per_numa();
            const unsigned tThreadsPerCore = Kokkos::hwloc::get_available_threads_per_core();
            std::cout << "P" << Comm::rank(aMachine) << ": hwloc { NUMA[" << tNumaCount << "]" << " CORE[" << tCoresPerNuma
                      << "]" << " PU[" << tThreadsPerCore << "] }" << std::endl;
        }
#if defined(KOKKOS_HAVE_CUDA)
        Plato::run_cuda_query(aMachine);
#endif
    }
    else
    {
        Plato::error_check(aProblem);
        ::Plato::driver(aOmegaHLib, aProblem, aMachine);
    }
}
// function run

}
// namespace Plato
