#include "Analyze_App.hpp"
#include <Teuchos_Comm.hpp>
#include <Teuchos_TimeMonitor.hpp>

#ifndef NDEBUG
#include <fenv.h>
#endif

void printTimingResultsMPMD(MPI_Comm &aComm)
{
  const std::string tTimerFilter = ""; //"Analyze:"; // Only timers beginning with this string get summarized.
  const bool tAlwaysWriteLocal = false;
  const bool tWriteGlobalStats = true;
  const bool tWriteZeroTimers  = false;
  Teuchos::RCP<const Teuchos::Comm<int> > tComm = Teuchos::rcp (new Teuchos::MpiComm<int> (aComm));
  std::ofstream tTimingOutputFileStream ("plato_analyze_timing_summary.txt", std::ofstream::out);
  Teuchos::TimeMonitor::summarize(tComm.ptr(), tTimingOutputFileStream, tAlwaysWriteLocal, 
                                  tWriteGlobalStats, tWriteZeroTimers, 
                                  Teuchos::ECounterSetOp::Intersection, tTimerFilter);
  tTimingOutputFileStream.close();
}

void safeExit(int aExitCode=0){
  Kokkos::finalize_all();
  MPI_Finalize();
  exit(aExitCode);
}

/******************************************************************************/
int main(int aArgc, char **aArgv)
/******************************************************************************/
{
#ifndef NDEBUG
//    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

    MPI_Init(&aArgc, &aArgv);
    Kokkos::initialize(aArgc, aArgv);

    Plato::Interface* tPlatoInterface = nullptr;
    try
    {
      tPlatoInterface = new Plato::Interface();
    }
    catch(...)
    {
      int tErrorCode = 1;
      safeExit(tErrorCode);
    }

    MPI_Comm tLocalComm;
    tPlatoInterface->getLocalComm(tLocalComm);

    Plato::MPMD_App* tMyApp = nullptr;
    try
    {
      tMyApp = new Plato::MPMD_App(aArgc, aArgv, tLocalComm);
    }
    catch(...)
    {
      tMyApp = nullptr;
      tPlatoInterface->Catch();
    }

    try
    {
      tPlatoInterface->registerPerformer(tMyApp);
    }
    catch(...)
    {
      int tErrorCode = 1;
      safeExit(tErrorCode);
    }

    try
    {
      tPlatoInterface->perform();
      printTimingResultsMPMD(tLocalComm);
    }
    catch(...)
    {
      safeExit();
    }

    if(tMyApp)
    {
      delete tMyApp;
    }
    
    if(tPlatoInterface)
    {
      delete tPlatoInterface;
    }

    safeExit();
}
