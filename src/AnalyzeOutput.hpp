#ifndef PLATO_OUTPUT_HPP
#define PLATO_OUTPUT_HPP

#include <string>

#include <Teuchos_ParameterList.hpp>

#include <Omega_h_tag.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_assoc.hpp>
#include <Omega_h_teuchos.hpp>

#include "PlatoUtilities.hpp"
#include "OmegaHUtilities.hpp"
#include "PlatoProblemFactory.hpp"
namespace Plato
{


/******************************************************************************//**
 * \brief Output data associated with an electro-mechanical simulation.
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aSolution        global solution data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
template<const Plato::OrdinalType SpatialDim>
inline void
electromechanical_output(
    const std::string     & aOutputFilePath,
    const Plato::Solution & aSolution,
    const Plato::DataMap  & aStateDataMap,
          Omega_h::Mesh   & aMesh
)
{
    const Plato::OrdinalType tTIME_STEP_INDEX = 0;
    auto tSubView = Kokkos::subview(aSolution.State, tTIME_STEP_INDEX, Kokkos::ALL());

    auto tNumVertices = aMesh.nverts();
    auto tNumDisp = tNumVertices * SpatialDim;
    constexpr auto tNumDispPerNode = SpatialDim;
    Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
    constexpr auto tNumDofsPerNode = SpatialDim + static_cast<Plato::OrdinalType>(1);
    Plato::copy<tNumDofsPerNode, tNumDispPerNode>(/*tStride=*/0, tNumVertices, tSubView, tDisp);

    auto tNumPot = tNumVertices;
    Omega_h::Write<Omega_h::Real> tPot(tNumPot, "Potential");
    constexpr auto tNumPotDofPerNode = static_cast<Plato::OrdinalType>(1);
    Plato::copy<tNumDofsPerNode, tNumPotDofPerNode>(/*tStride=*/SpatialDim, tNumVertices, tSubView, tPot);

    const Plato::Scalar tRestartTime = 0.;
    Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

    aMesh.add_tag(Omega_h::VERT, "Displacements", SpatialDim, Omega_h::Reals(tDisp));
    aMesh.add_tag(Omega_h::VERT, "Potential", 1, Omega_h::Reals(tPot));
    Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
    tWriter.write(/*time_index*/1, /*current_time=*/1.0, tTags);
}
// function electromechanical_output


/******************************************************************************//**
 * \brief Output data associated with a stabilized thermo-mechanical simulation.
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aSolution        global solution data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
template<const Plato::OrdinalType SpatialDim>
inline void
stabilized_thermomechanical_output(
    const std::string     & aOutputFilePath,
    const Plato::Solution & aSolution,
    const Plato::DataMap  & aStateDataMap,
          Omega_h::Mesh   & aMesh
)
{
    auto tState = aSolution.State;

    const Plato::Scalar tRestartTime = 0.;
    Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

    auto tNumSteps = tState.extent(0);
    for(decltype(tNumSteps) tStepIndex=0; tStepIndex<tNumSteps; tStepIndex++)
    {
        auto tSubView = Kokkos::subview(tState, tStepIndex, Kokkos::ALL());

        auto tNumVertices = aMesh.nverts();
        Omega_h::Write<Omega_h::Real> tTemp(tNumVertices, "Temperature");
        constexpr auto tNumTempPerNode = static_cast<Plato::OrdinalType>(1);
        constexpr auto tNumDofsPerNode = SpatialDim + static_cast<Plato::OrdinalType>(2);
        Plato::copy<tNumDofsPerNode, tNumTempPerNode> (/*tStride=*/ SpatialDim+1, tNumVertices, tSubView, tTemp);

        Omega_h::Write<Omega_h::Real> tPress(tNumVertices, "Pressure");
        constexpr auto tNumPressPerNode = static_cast<Plato::OrdinalType>(1);
        Plato::copy<tNumDofsPerNode, tNumPressPerNode> (/*tStride=*/ SpatialDim, tNumVertices, tSubView, tPress);

        auto tNumDisp = tNumVertices * SpatialDim;
        constexpr auto tNumDispPerNode = SpatialDim;
        Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
        Plato::copy<tNumDofsPerNode, tNumDispPerNode>(/*tStride=*/0, tNumVertices, tSubView, tDisp);


        aMesh.add_tag(Omega_h::VERT, "Displacements", tNumDispPerNode,  Omega_h::Reals(tDisp));
        aMesh.add_tag(Omega_h::VERT, "Pressure",      tNumPressPerNode, Omega_h::Reals(tPress));
        aMesh.add_tag(Omega_h::VERT, "Temperature",   tNumTempPerNode,  Omega_h::Reals(tTemp));

        Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
        tWriter.write(/*time_index*/tStepIndex, /*current_time=*/(Plato::Scalar)tStepIndex, tTags);
   }
}
// function stabilized_thermomechanical_output


/******************************************************************************//**
 * \brief Output data associated with a thermo-mechanical simulation.
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aSolution        global solution data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
template<const Plato::OrdinalType SpatialDim>
inline void
thermomechanical_output(
    const std::string     & aOutputFilePath,
    const Plato::Solution & aSolution,
    const Plato::DataMap  & aStateDataMap,
          Omega_h::Mesh   & aMesh
)
{
    auto tState = aSolution.State;

    const Plato::Scalar tRestartTime = 0.;
    Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

    auto tNumSteps = tState.extent(0);
    for(decltype(tNumSteps) tStepIndex=0; tStepIndex<tNumSteps; tStepIndex++)
    {
        auto tSubView = Kokkos::subview(tState, tStepIndex, Kokkos::ALL());

        auto tNumVertices = aMesh.nverts();
        Omega_h::Write<Omega_h::Real> tTemp(tNumVertices, "Temperature");
        constexpr auto tNumTempPerNode = static_cast<Plato::OrdinalType>(1);
        constexpr auto tNumDofsPerNode = SpatialDim + static_cast<Plato::OrdinalType>(1);
        Plato::copy<tNumDofsPerNode, tNumTempPerNode> (/*tStride=*/ SpatialDim, tNumVertices, tSubView, tTemp);

        auto tNumDisp = tNumVertices * SpatialDim;
        constexpr auto tNumDispPerNode = SpatialDim;
        Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
        Plato::copy<tNumDofsPerNode, tNumDispPerNode>(/*tStride=*/0, tNumVertices, tSubView, tDisp);

        aMesh.add_tag(Omega_h::VERT, "Displacements", tNumDispPerNode, Omega_h::Reals(tDisp));
        aMesh.add_tag(Omega_h::VERT, "Temperature",   tNumTempPerNode, Omega_h::Reals(tTemp));

        if (aStateDataMap.stateDataMaps.size() > tStepIndex)
        {
            Plato::add_element_state_tags(aMesh, aStateDataMap, tStepIndex);
        }

        Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
        tWriter.write(/*time_index*/tStepIndex, /*current_time=*/(Plato::Scalar)tStepIndex, tTags);
    }
}
// function thermomechanical_output


/******************************************************************************//**
 * \brief Output data associated with a mechanical simulation.
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aSolution        global solution data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
template<const Plato::OrdinalType SpatialDim>
inline void
mechanical_output(
    const std::string     & aOutputFilePath,
    const Plato::Solution & aSolution,
    const Plato::DataMap  & aStateDataMap,
          Omega_h::Mesh   & aMesh)
{
    auto tState = aSolution.State;
    auto tStateDot = aSolution.StateDot;
    auto tStateDotDot = aSolution.StateDotDot;

    const Plato::Scalar tRestartTime = 0.;
    Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

    auto tNumSteps = tState.extent(0);
    for(decltype(tNumSteps) tStepIndex=0; tStepIndex<tNumSteps; tStepIndex++)
    {
        if (tState.extent(0) != 0)
        {
            auto tSubView = Kokkos::subview(tState, tStepIndex, Kokkos::ALL());
            auto tNumVertices = aMesh.nverts();
            auto tNumDisp = tNumVertices * SpatialDim;
            constexpr auto tNumDofsPerNode = SpatialDim;
            constexpr auto tNumDispPerNode = SpatialDim;
            Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
            Plato::copy<tNumDofsPerNode, tNumDispPerNode>(/*stride=*/0, tNumVertices, tSubView, tDisp);
            aMesh.add_tag(Omega_h::VERT, "Displacements", tNumDispPerNode, Omega_h::Reals(tDisp));
        }

        if (tStateDot.extent(0) != 0)
        {
            auto tSubView = Kokkos::subview(tStateDot, tStepIndex, Kokkos::ALL());
            auto tNumVertices = aMesh.nverts();
            auto tNumVel = tNumVertices * SpatialDim;
            constexpr auto tNumDofsPerNode = SpatialDim;
            constexpr auto tNumVelPerNode = SpatialDim;
            Omega_h::Write<Omega_h::Real> tVel(tNumVel, "Velocity");
            Plato::copy<tNumDofsPerNode, tNumVelPerNode>(/*stride=*/0, tNumVertices, tSubView, tVel);
            aMesh.add_tag(Omega_h::VERT, "Velocity", tNumVelPerNode, Omega_h::Reals(tVel));
        }

        if (tStateDotDot.extent(0) != 0)
        {
            auto tSubView = Kokkos::subview(tStateDotDot, tStepIndex, Kokkos::ALL());
            auto tNumVertices = aMesh.nverts();
            auto tNumAcc = tNumVertices * SpatialDim;
            constexpr auto tNumDofsPerNode = SpatialDim;
            constexpr auto tNumAccPerNode = SpatialDim;
            Omega_h::Write<Omega_h::Real> tAcc(tNumAcc, "Acceleration");
            Plato::copy<tNumDofsPerNode, tNumAccPerNode>(/*stride=*/0, tNumVertices, tSubView, tAcc);
            aMesh.add_tag(Omega_h::VERT, "Acceleration", tNumAccPerNode, Omega_h::Reals(tAcc));
        }

        if (aStateDataMap.stateDataMaps.size() > tStepIndex)
        {
            Plato::add_element_state_tags(aMesh, aStateDataMap, tStepIndex);
        }
        Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
        tWriter.write(/*time_index*/tStepIndex, /*current_time=*/(Plato::Scalar)tStepIndex, tTags);
    }
}
// function mechanical_output


/******************************************************************************//**
 * \brief Output data associated with a stabilized mechanical simulation.
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aSolution        global solution data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
template<const Plato::OrdinalType SpatialDim>
inline void
stabilized_mechanical_output(
    const std::string     & aOutputFilePath,
    const Plato::Solution & aSolution,
    const Plato::DataMap  & aStateDataMap,
          Omega_h::Mesh   & aMesh)
{
    auto tState = aSolution.State;

    const Plato::Scalar tRestartTime = 0.;
    Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

    auto tNumSteps = tState.extent(0);
    for(decltype(tNumSteps) tStepIndex=0; tStepIndex<tNumSteps; tStepIndex++)
    {
        auto tSubView = Kokkos::subview(tState, tStepIndex, Kokkos::ALL());

        auto tNumVertices = aMesh.nverts();
        Omega_h::Write<Omega_h::Real> tPress(tNumVertices, "Pressure");
        constexpr auto tNumPressPerNode = static_cast<Plato::OrdinalType>(1);
        constexpr auto tNumDofsPerNode = SpatialDim + static_cast<Plato::OrdinalType>(1);
        Plato::copy<tNumDofsPerNode, tNumPressPerNode> (/*tStride=*/ SpatialDim, tNumVertices, tSubView, tPress);

        auto tNumDisp = tNumVertices * SpatialDim;
        constexpr auto tNumDispPerNode = SpatialDim;
        Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
        Plato::copy<tNumDofsPerNode, tNumDispPerNode>(/*tStride=*/0, tNumVertices, tSubView, tDisp);

        aMesh.add_tag(Omega_h::VERT, "Displacements", tNumDispPerNode,  Omega_h::Reals(tDisp));
        aMesh.add_tag(Omega_h::VERT, "Pressure",      tNumPressPerNode, Omega_h::Reals(tPress));

        if (aStateDataMap.stateDataMaps.size() > tStepIndex)
        {
            Plato::add_element_state_tags(aMesh, aStateDataMap, tStepIndex);
        }
        Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
        tWriter.write(/*time_index*/tStepIndex, /*current_time=*/(Plato::Scalar)tStepIndex, tTags);
    }
}
// function stabilized_mechanical_output


/******************************************************************************//**
 * \brief Output data associated with a thermal simulation.
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aSolution        global solution data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
template<const Plato::OrdinalType SpatialDim>
inline void
thermal_output(
    const std::string     & aOutputFilePath,
    const Plato::Solution & aSolution,
    const Plato::DataMap  & aStateDataMap,
          Omega_h::Mesh   & aMesh)
{
    auto tState = aSolution.State;

    const Plato::Scalar tRestartTime = 0.;
    Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

    auto tNumSteps = tState.extent(0);
    for(decltype(tNumSteps) tStepIndex=0; tStepIndex<tNumSteps; tStepIndex++)
    {
        auto tSubView = Kokkos::subview(tState, tStepIndex, Kokkos::ALL());

        auto tNumVertices = aMesh.nverts();
        Omega_h::Write<Omega_h::Real> tTemp(tNumVertices, "Temperature");

        const Plato::OrdinalType tStride = 0;
        constexpr auto tNumTempPerNode = static_cast<Plato::OrdinalType>(1);
        constexpr auto tNumDofsPerNode = static_cast<Plato::OrdinalType>(1);
        Plato::copy<tNumDofsPerNode, tNumTempPerNode>(tStride, tNumVertices, tSubView, tTemp);

        aMesh.add_tag(Omega_h::VERT, "Temperature", tNumTempPerNode, Omega_h::Reals(tTemp));

        if (aStateDataMap.stateDataMaps.size() > tStepIndex)
        {
            Plato::add_element_state_tags(aMesh, aStateDataMap, tStepIndex);
        }
        Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
        tWriter.write(/*time_index*/tStepIndex, /*current_time=*/(Plato::Scalar)tStepIndex, tTags);
    }
}
// function thermal_output


/******************************************************************************//**
 * \brief Output data associated with a Plato Analyze simulation.
 * \param [in] aInputData       input parameters list
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aState           global state data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
template<const Plato::OrdinalType SpatialDim>
void
output(
          Teuchos::ParameterList & aInputData,
    const std::string            & aOutputFilePath,
    const Plato::Solution        & aSolution,
    const Plato::DataMap         & aStateDataMap,
          Omega_h::Mesh          & aMesh
)
{
    auto tProblemSpecs = aInputData.sublist("Plato Problem");
    assert(tProblemSpecs.isParameter("Physics"));
    auto tPhysics = tProblemSpecs.get < std::string > ("Physics");
    auto tPDE     = tProblemSpecs.get < std::string > ("PDE Constraint");

    if(tPhysics == "Electromechanical")
    {
        Plato::electromechanical_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
    }
    else
    if(tPhysics == "Thermomechanical")
    {
        if( tPDE == "Stabilized Elliptic" )
        {
            Plato::stabilized_thermomechanical_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
        }
        else
        {
            Plato::thermomechanical_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
        }
    }
    else
    if(tPhysics == "Mechanical")
    {
        if(tPDE == "Infinite Strain Plasticity")
        {
            Plato::stabilized_mechanical_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
        }
        else
        {
            Plato::mechanical_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
        }
    }
    else
    if(tPhysics == "Stabilized Mechanical")
    {
        Plato::stabilized_mechanical_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
    }
    else
    if(tPhysics == "StructuralDynamics")
    {
    }
    else
    if(tPhysics == "Thermal")
    {
        Plato::thermal_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
    }
}
// function output


/******************************************************************************//**
 * \brief Write data associated with a Plato Analyze simulation to an output file.
 * \param [in] aInputData       input parameters list
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aSolution        global solution data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
inline void
write(
          Teuchos::ParameterList & aInputData,
    const std::string            & aOutputFilePath,
    const Plato::Solution        & aSolution,
    const Plato::ScalarVector    & aControl,
    const Plato::DataMap         & aStateDataMap,
          Omega_h::Mesh          & aMesh
)
{
    auto tNumVertices = aMesh.nverts();
    Omega_h::Write<Omega_h::Real> tControl (tNumVertices, "Control");
    Plato::copy<1, 1>(/*tStride=*/0, tNumVertices, aControl, tControl);
    aMesh.add_tag(Omega_h::VERT, "Control", 1, Omega_h::Reals(tControl));

    const Plato::OrdinalType tSpaceDim = aInputData.get<Plato::OrdinalType>("Spatial Dimension", 3);
    if(tSpaceDim == static_cast<Plato::OrdinalType>(1))
    {
        #ifdef PLATOANALYZE_1D
        Plato::output<1>(aInputData, aOutputFilePath, aSolution, aStateDataMap, aMesh);
        #else
        throw std::runtime_error("1D physics is not compiled.");
        #endif
    } else
    if(tSpaceDim == static_cast<Plato::OrdinalType>(2))
    {
        #ifdef PLATOANALYZE_2D
        Plato::output<2>(aInputData, aOutputFilePath, aSolution, aStateDataMap, aMesh);
        #else
        throw std::runtime_error("2D physics is not compiled.");
        #endif
    } else
    if(tSpaceDim == static_cast<Plato::OrdinalType>(3))
    {
        #ifdef PLATOANALYZE_3D
        Plato::output<3>(aInputData, aOutputFilePath, aSolution, aStateDataMap, aMesh);
        #else
        throw std::runtime_error("3D physics is not compiled.");
        #endif
    }
}
// function write


}
// namespace Plato

#endif
