#ifndef PLATO_OUTPUT_HPP
#define PLATO_OUTPUT_HPP

#include <string>

#include <Teuchos_ParameterList.hpp>

#include <Omega_h_tag.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_assoc.hpp>

#include "Solutions.hpp"
#include "UtilsOmegaH.hpp"
#include "PlatoUtilities.hpp"

namespace Plato
{

    /******************************************************************************/ /**
 * \brief Output data associated with an electro-mechanical simulation.
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aSolution        global solution data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
    template <const Plato::OrdinalType SpatialDim>
    inline void
    electromechanical_output(
        const std::string &aOutputFilePath,
        const Plato::Solutions &aSolution,
        const Plato::DataMap &aStateDataMap,
        Omega_h::Mesh &aMesh)
    {
        auto tTags = aSolution.tags();
        for (auto &tTag : tTags)
        {
            auto tState = aSolution.get(tTag);
            const Plato::OrdinalType tTIME_STEP_INDEX = 0;
            auto tSubView = Kokkos::subview(tState, tTIME_STEP_INDEX, Kokkos::ALL());

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
            tWriter.write(/*time_index*/ 1, /*current_time=*/1.0, tTags);
        }
    }
    // function electromechanical_output

    /******************************************************************************/ /**
 * \brief Output data associated with a stabilized thermo-mechanical simulation.
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aSolution        global solution data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
    template <const Plato::OrdinalType SpatialDim>
    inline void
    stabilized_thermomechanical_output(
        const std::string &aOutputFilePath,
        const Plato::Solutions &aSolution,
        const Plato::DataMap &aStateDataMap,
        Omega_h::Mesh &aMesh)
    {
        const Plato::Scalar tRestartTime = 0.;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

        auto tTags = aSolution.tags();
        for (auto &tTag : tTags)
        {
            auto tState = aSolution.get(tTag);
            auto tNumSteps = tState.extent(0);
            for (decltype(tNumSteps) tStepIndex = 0; tStepIndex < tNumSteps; tStepIndex++)
            {
                auto tSubView = Kokkos::subview(tState, tStepIndex, Kokkos::ALL());

                auto tNumVertices = aMesh.nverts();
                Omega_h::Write<Omega_h::Real> tTemp(tNumVertices, "Temperature");
                constexpr auto tNumTempPerNode = static_cast<Plato::OrdinalType>(1);
                constexpr auto tNumDofsPerNode = SpatialDim + static_cast<Plato::OrdinalType>(2);
                Plato::copy<tNumDofsPerNode, tNumTempPerNode>(/*tStride=*/SpatialDim + 1, tNumVertices, tSubView, tTemp);

                Omega_h::Write<Omega_h::Real> tPress(tNumVertices, "Pressure");
                constexpr auto tNumPressPerNode = static_cast<Plato::OrdinalType>(1);
                Plato::copy<tNumDofsPerNode, tNumPressPerNode>(/*tStride=*/SpatialDim, tNumVertices, tSubView, tPress);

                auto tNumDisp = tNumVertices * SpatialDim;
                constexpr auto tNumDispPerNode = SpatialDim;
                Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
                Plato::copy<tNumDofsPerNode, tNumDispPerNode>(/*tStride=*/0, tNumVertices, tSubView, tDisp);

                aMesh.add_tag(Omega_h::VERT, "Displacements", tNumDispPerNode, Omega_h::Reals(tDisp));
                aMesh.add_tag(Omega_h::VERT, "Pressure", tNumPressPerNode, Omega_h::Reals(tPress));
                aMesh.add_tag(Omega_h::VERT, "Temperature", tNumTempPerNode, Omega_h::Reals(tTemp));

                Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
                tWriter.write(/*time_index*/ tStepIndex, /*current_time=*/(Plato::Scalar)tStepIndex, tTags);
            }
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
    template <const Plato::OrdinalType SpatialDim>
    inline void
    thermomechanical_output(
        const std::string &aOutputFilePath,
        const Plato::Solutions &aSolution,
        const Plato::DataMap &aStateDataMap,
        Omega_h::Mesh &aMesh)
    {
        const Plato::Scalar tRestartTime = 0.;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

        auto tTags = aSolution.tags();
        for (auto &tTag : tTags)
        {
            auto tState = aSolution.get(tTag);
            auto tNumSteps = tState.extent(0);
            for (decltype(tNumSteps) tStepIndex = 0; tStepIndex < tNumSteps; tStepIndex++)
            {
                auto tSubView = Kokkos::subview(tState, tStepIndex, Kokkos::ALL());

                auto tNumVertices = aMesh.nverts();
                Omega_h::Write<Omega_h::Real> tTemp(tNumVertices, "Temperature");
                constexpr auto tNumTempPerNode = static_cast<Plato::OrdinalType>(1);
                constexpr auto tNumDofsPerNode = SpatialDim + static_cast<Plato::OrdinalType>(1);
                Plato::copy<tNumDofsPerNode, tNumTempPerNode>(/*tStride=*/SpatialDim, tNumVertices, tSubView, tTemp);

                auto tNumDisp = tNumVertices * SpatialDim;
                constexpr auto tNumDispPerNode = SpatialDim;
                Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
                Plato::copy<tNumDofsPerNode, tNumDispPerNode>(/*tStride=*/0, tNumVertices, tSubView, tDisp);

                aMesh.add_tag(Omega_h::VERT, "Displacements", tNumDispPerNode, Omega_h::Reals(tDisp));
                aMesh.add_tag(Omega_h::VERT, "Temperature", tNumTempPerNode, Omega_h::Reals(tTemp));

                if (aStateDataMap.stateDataMaps.size() > tStepIndex)
                {
                    Plato::omega_h::add_state_tags(aMesh, aStateDataMap, tStepIndex);
                }

                Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
                tWriter.write(/*time_index*/ tStepIndex, /*current_time=*/(Plato::Scalar)tStepIndex, tTags);
            }
        }
    }
    // function thermomechanical_output

    /******************************************************************************/ /**
 * \brief Output data associated with a mechanical simulation.
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aSolution        global solution data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
    template <const Plato::OrdinalType SpatialDim>
    inline void
    mechanical_output(
        const std::string &aOutputFilePath,
        const Plato::Solutions &aSolution,
        const Plato::DataMap &aStateDataMap,
        Omega_h::Mesh &aMesh)
    {
        const Plato::Scalar tRestartTime = 0.;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

        auto tTags = aSolution.tags();
        for (auto &tTag : tTags)
        {
            auto tState = aSolution.get(tTag);
            auto tNumSteps = tState.extent(0);
            for (decltype(tNumSteps) tStepIndex = 0; tStepIndex < tNumSteps; tStepIndex++)
            {
                auto tSubView = Kokkos::subview(tState, tStepIndex, Kokkos::ALL());
                auto tNumVertices = aMesh.nverts();
                auto tNumState = tNumVertices * SpatialDim;
                constexpr auto tNumDofsPerNode = SpatialDim;
                constexpr auto tNumStatePerNode = SpatialDim;
                Omega_h::Write<Omega_h::Real> tStateData(tNumState, "State");
                Plato::copy<tNumDofsPerNode, tNumStatePerNode>(/*stride=*/0, tNumVertices, tSubView, tStateData);
                aMesh.add_tag(Omega_h::VERT, "Displacements", tNumStatePerNode, Omega_h::Reals(tStateData));

                if (aStateDataMap.stateDataMaps.size() > tStepIndex)
                {
                    Plato::omega_h::add_state_tags(aMesh, aStateDataMap, tStepIndex);
                }

                Omega_h::TagSet tTagsOmegaH = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
                tWriter.write(/*time_index*/ tStepIndex, /*current_time=*/(Plato::Scalar)tStepIndex, tTagsOmegaH);
            }
        }
    }
    // function mechanical_output

    /******************************************************************************/ /**
 * \brief Output data associated with a stabilized mechanical simulation.
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aSolution        global solution data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
    template <const Plato::OrdinalType SpatialDim>
    inline void
    stabilized_mechanical_output(
        const std::string &aOutputFilePath,
        const Plato::Solutions &aSolution,
        const Plato::DataMap &aStateDataMap,
        Omega_h::Mesh &aMesh)
    {
        const Plato::Scalar tRestartTime = 0.;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

        auto tTags = aSolution.tags();
        for (auto &tTag : tTags)
        {
            auto tState = aSolution.get(tTag);
            auto tNumSteps = tState.extent(0);
            for (decltype(tNumSteps) tStepIndex = 0; tStepIndex < tNumSteps; tStepIndex++)
            {
                auto tSubView = Kokkos::subview(tState, tStepIndex, Kokkos::ALL());

                auto tNumVertices = aMesh.nverts();
                Omega_h::Write<Omega_h::Real> tPress(tNumVertices, "Pressure");
                constexpr auto tNumPressPerNode = static_cast<Plato::OrdinalType>(1);
                constexpr auto tNumDofsPerNode = SpatialDim + static_cast<Plato::OrdinalType>(1);
                Plato::copy<tNumDofsPerNode, tNumPressPerNode>(/*tStride=*/SpatialDim, tNumVertices, tSubView, tPress);

                auto tNumDisp = tNumVertices * SpatialDim;
                constexpr auto tNumDispPerNode = SpatialDim;
                Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
                Plato::copy<tNumDofsPerNode, tNumDispPerNode>(/*tStride=*/0, tNumVertices, tSubView, tDisp);

                aMesh.add_tag(Omega_h::VERT, "Displacements", tNumDispPerNode, Omega_h::Reals(tDisp));
                aMesh.add_tag(Omega_h::VERT, "Pressure", tNumPressPerNode, Omega_h::Reals(tPress));

                if (aStateDataMap.stateDataMaps.size() > tStepIndex)
                {
                    Plato::omega_h::add_state_tags(aMesh, aStateDataMap, tStepIndex);
                }
                Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
                tWriter.write(/*time_index*/ tStepIndex, /*current_time=*/(Plato::Scalar)tStepIndex, tTags);
            }
        }
    }
    // function stabilized_mechanical_output

    /******************************************************************************/ /**
 * \brief Output data associated with a thermal simulation.
 * \param [in] aOutputFilePath  output viz file path
 * \param [in] aSolution        global solution data
 * \param [in] aStateDataMap    Plato Analyze data map
 * \param [in] aMesh            mesh database
 **********************************************************************************/
    template <const Plato::OrdinalType SpatialDim>
    inline void
    thermal_output(
        const std::string      & aOutputFilePath,
        const Plato::Solutions & aSolution,
        const Plato::DataMap   & aStateDataMap,
              Omega_h::Mesh    & aMesh)
    {
        const Plato::Scalar tRestartTime = 0.;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

        auto tTags = aSolution.tags();
        for (auto &tTag : tTags)
        {
            auto tState = aSolution.get(tTag);
            auto tNumSteps = tState.extent(0);
            for (decltype(tNumSteps) tStepIndex = 0; tStepIndex < tNumSteps; tStepIndex++)
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
                    Plato::omega_h::add_state_tags(aMesh, aStateDataMap, tStepIndex);
                }
                Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
                tWriter.write(/*time_index*/ tStepIndex, /*current_time=*/(Plato::Scalar)tStepIndex, tTags);
            }
        }
    }
    // function thermal_output

    /******************************************************************************/ /**
    * \brief Output data for all your output needs
    * \param [in] aOutputFilePath  output viz file path
    * \param [in] aSolutionsOutput global solution data for output
    * \param [in] aStateDataMap    Plato Analyze data map
    * \param [in] aMesh            mesh database
    **********************************************************************************/
    template <const Plato::OrdinalType SpatialDim>
    inline void
    universal_solution_output(
        const std::string      & aOutputFilePath,
        const Plato::Solutions & aSolutionsOutput,
        const Plato::DataMap   & aStateDataMap,
              Omega_h::Mesh    & aMesh)
    {
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, 0.0);
        auto tNumTimeSteps = aSolutionsOutput.getNumTimeSteps();
        for (Plato::OrdinalType tStepIndex = 0; tStepIndex < tNumTimeSteps; ++tStepIndex)
        {
            for (auto &tSolutionOutputTag : aSolutionsOutput.tags())
            {
                auto tSolutions = aSolutionsOutput.get(tSolutionOutputTag);
                auto tNumDofs = aSolutionsOutput.getNumDofs(tSolutionOutputTag);
                Plato::ScalarVector tSolution = Kokkos::subview(tSolutions, tStepIndex, Kokkos::ALL());
                Omega_h::Write<Omega_h::Real> tSolutionOmegaH(tSolution.size(), "OmegaH Solution");
                Plato::copy_1Dview_to_write(tSolution, tSolutionOmegaH);
                if (aStateDataMap.stateDataMaps.size() > tStepIndex)
                {
                    Plato::omega_h::add_state_tags(aMesh, aStateDataMap, tStepIndex);
                }
                aMesh.add_tag(Omega_h::VERT, tSolutionOutputTag, tNumDofs, Omega_h::Reals(tSolutionOmegaH));
            }
            Omega_h::TagSet tTagsOmegaH = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
            tWriter.write(/*time_index*/ tStepIndex, /*current_time=*/(Plato::Scalar)tStepIndex, tTagsOmegaH);
        }
    }

    /******************************************************************************/ /**
    * \brief Output data associated with a Plato Analyze simulation.
    * \param [in] aOutputFilePath  output viz file path
    * \param [in] aState           global state data
    * \param [in] aStateDataMap    Plato Analyze data map
    * \param [in] aMesh            mesh database
    **********************************************************************************/
    template 
    <const Plato::OrdinalType SpatialDim>
    void output
    (const std::string      &aOutputFilePath,
     const Plato::Solutions &aSolution,
     const Plato::DataMap   &aStateDataMap,
           Omega_h::Mesh    &aMesh)
    {
        auto tPDE = aSolution.pde();
        auto tPhysics = aSolution.physics();

        auto tLowerPDE = Plato::tolower(tPDE);
        auto tLowerPhysics = Plato::tolower(tPhysics);
        if (tLowerPhysics == "electromechanical")
        {
            Plato::electromechanical_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
        }
        else 
        if (tLowerPhysics == "thermomechanical")
        {
            if (tLowerPDE == "stabilized elliptic")
            {
                Plato::stabilized_thermomechanical_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
            }
            else
            {
                Plato::thermomechanical_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
            }
        }
        else 
        if (tLowerPhysics == "mechanical")
        {
            Plato::mechanical_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
        }
        else 
        if (tLowerPhysics == "plasticity")
        {
            Plato::stabilized_mechanical_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
        }
        else 
        if (tLowerPhysics == "stabilized mechanical")
        {
            Plato::stabilized_mechanical_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
        }
        else 
        if (tLowerPhysics == "structuraldynamics")
        {
        }
        else 
        if (tLowerPhysics == "thermal")
        {
            Plato::thermal_output<SpatialDim>(aOutputFilePath, aSolution, aStateDataMap, aMesh);
        }
    }
    // function output

}
// namespace Plato

#endif
