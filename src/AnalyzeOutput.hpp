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
#include "PlatoProblemFactory.hpp"
namespace Plato
{

inline void addElementStateTags(Omega_h::Mesh& aMesh, const Plato::DataMap& aStateDataMap, int aStepIndex)
{ 
    auto tDataMap = aStateDataMap.getState(aStepIndex);

    auto tNumElements = aMesh.nelems();
    {   // ScalarVectors
        //
        auto& tVars = tDataMap.scalarVectors;
        for(auto tVar=tVars.begin(); tVar!=tVars.end(); ++tVar)
        {
            auto& tElemStateName = tVar->first;
            auto& tElemStateData = tVar->second;
            if(tElemStateData.extent(0) == tNumElements)
            {
                Omega_h::Write<Omega_h::Real> tElemStateWrite(tNumElements, tElemStateName);
                Plato::copy_1Dview_to_write(tElemStateData, tElemStateWrite);
                aMesh.add_tag(aMesh.dim(), tElemStateName, /*numDataPerElement=*/1, Omega_h::Reals(tElemStateWrite));
            }
        }
    }
    {   // ScalarMultiVectors
        //
        auto& tVars = tDataMap.scalarMultiVectors;
        for(auto tVar=tVars.begin(); tVar!=tVars.end(); ++tVar)
        {
            auto& tElemStateName = tVar->first;
            auto& tElemStateData = tVar->second;
            if(tElemStateData.extent(0) == tNumElements)
            {
                auto tNumDataPerElement = tElemStateData.extent(1);
                auto tNumData = tNumElements * tNumDataPerElement;
                Omega_h::Write<Omega_h::Real> tElemStateWrite(tNumData, tElemStateName);
                Plato::copy_2Dview_to_write(tElemStateData, tElemStateWrite);
                aMesh.add_tag(aMesh.dim(), tElemStateName, tNumDataPerElement, Omega_h::Reals(tElemStateWrite));
            }
        }
    }
}

template<const Plato::OrdinalType SpatialDim>
void output(Teuchos::ParameterList & aParamList,
            const std::string & aOutputFilePath,
            const Plato::ScalarMultiVector & aState,
            const Plato::DataMap & aStateDataMap,
            Omega_h::Mesh& aMesh)
{
    auto tProblemSpecs = aParamList.sublist("Plato Problem");
    assert(tProblemSpecs.isParameter("Physics"));
    auto tPhysics = tProblemSpecs.get < std::string > ("Physics");
    auto tPDE     = tProblemSpecs.get < std::string > ("PDE Constraint");

    if(tPhysics == "Electromechanical")
    {
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());

        auto tNumVertices = aMesh.nverts();
        auto tNumDisp = tNumVertices * SpatialDim;
        Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
        Plato::copy<SpatialDim+1, SpatialDim>(/*tStride=*/0, tNumVertices, tSubView, tDisp);

        auto tNumPot  = tNumVertices;
        Omega_h::Write<Omega_h::Real> tPot (tNumPot, "Potential");
        Plato::copy<SpatialDim+1, 1>(/*tStride=*/SpatialDim, tNumVertices, tSubView, tPot);

        const Plato::Scalar tRestartTime = 0.;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

        aMesh.add_tag(Omega_h::VERT, "Displacements", SpatialDim , Omega_h::Reals(tDisp));
        aMesh.add_tag(Omega_h::VERT, "Potential",     1          , Omega_h::Reals(tPot));
        Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
        tWriter.write(/*time_index*/1, /*current_time=*/1.0, tTags);
    } else
    if(tPhysics == "Thermomechanical")
    {
        const Plato::Scalar tRestartTime = 0.;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);
 
        if( tPDE == "Stabilized Elliptic" )
        {
            auto nSteps = aState.extent(0);
            for(decltype(nSteps) iStep=0; iStep<nSteps; iStep++){
              auto tSubView = Kokkos::subview(aState, iStep, Kokkos::ALL());

              auto tNumVertices = aMesh.nverts();
              Omega_h::Write<Omega_h::Real> tTemp(tNumVertices, "Temperature");
              Plato::copy<SpatialDim+2 /*input_num_dof_per_node*/, 1 /*output_num_dof_per_node*/> (/*tStride=*/ SpatialDim+1, tNumVertices, tSubView, tTemp);


              Omega_h::Write<Omega_h::Real> tPress(tNumVertices, "Pressure");
              Plato::copy<SpatialDim+2 /*input_num_dof_per_node*/, 1 /*output_num_dof_per_node*/> (/*tStride=*/ SpatialDim, tNumVertices, tSubView, tPress);

              auto tNumDisp = tNumVertices * SpatialDim;
              Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
              Plato::copy<SpatialDim+2, SpatialDim>(/*tStride=*/0, tNumVertices, tSubView, tDisp);


              aMesh.add_tag(Omega_h::VERT, "Displacements", SpatialDim,                    Omega_h::Reals(tDisp));
              aMesh.add_tag(Omega_h::VERT, "Pressure",      1 /*output_num_dof_per_node*/, Omega_h::Reals(tPress));
              aMesh.add_tag(Omega_h::VERT, "Temperature",   1 /*output_num_dof_per_node*/, Omega_h::Reals(tTemp));

              Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
              tWriter.write(/*time_index*/iStep, /*current_time=*/(Plato::Scalar)iStep, tTags);
           }
        } else {
            auto nSteps = aState.extent(0);
            for(decltype(nSteps) iStep=0; iStep<nSteps; iStep++){
              auto tSubView = Kokkos::subview(aState, iStep, Kokkos::ALL());

              auto tNumVertices = aMesh.nverts();
              Omega_h::Write<Omega_h::Real> tTemp(tNumVertices, "Temperature");
              Plato::copy<SpatialDim+1 /*input_num_dof_per_node*/, 1 /*output_num_dof_per_node*/> (/*tStride=*/ SpatialDim, tNumVertices, tSubView, tTemp);

              auto tNumDisp = tNumVertices * SpatialDim;
              Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
              Plato::copy<SpatialDim+1, SpatialDim>(/*tStride=*/0, tNumVertices, tSubView, tDisp);

              aMesh.add_tag(Omega_h::VERT, "Displacements", SpatialDim,                  Omega_h::Reals(tDisp));
              aMesh.add_tag(Omega_h::VERT, "Temperature", 1 /*output_num_dof_per_node*/, Omega_h::Reals(tTemp));

              Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
              tWriter.write(/*time_index*/iStep, /*current_time=*/(Plato::Scalar)iStep, tTags);
            }
        }
    } else
    if(tPhysics == "Mechanical")
    {
        const Plato::Scalar tRestartTime = 0.;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

        auto nSteps = aState.extent(0);
        for(decltype(nSteps) iStep=0; iStep<nSteps; iStep++){
          auto tSubView = Kokkos::subview(aState, iStep, Kokkos::ALL());

          auto tNumVertices = aMesh.nverts();
          auto tNumDisp = tNumVertices * SpatialDim;
          Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
          Plato::copy<SpatialDim, SpatialDim>(/*stride=*/0, tNumVertices, tSubView, tDisp);

          aMesh.add_tag(Omega_h::VERT, "Displacements", SpatialDim, Omega_h::Reals(tDisp));

          addElementStateTags(aMesh, aStateDataMap, iStep);
          Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
          tWriter.write(/*time_index*/iStep, /*current_time=*/(Plato::Scalar)iStep, tTags);
        }
    } else
    if(tPhysics == "Stabilized Mechanical")
    {
        const Plato::Scalar tRestartTime = 0.;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

        auto nSteps = aState.extent(0);
        for(decltype(nSteps) iStep=0; iStep<nSteps; iStep++){
          auto tSubView = Kokkos::subview(aState, iStep, Kokkos::ALL());

          auto tNumVertices = aMesh.nverts();
          Omega_h::Write<Omega_h::Real> tPress(tNumVertices, "Pressure");
          Plato::copy<SpatialDim+1 /*input_num_dof_per_node*/, 1 /*output_num_dof_per_node*/> (/*tStride=*/ SpatialDim, tNumVertices, tSubView, tPress);

          auto tNumDisp = tNumVertices * SpatialDim;
          Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
          Plato::copy<SpatialDim+1, SpatialDim>(/*tStride=*/0, tNumVertices, tSubView, tDisp);

          aMesh.add_tag(Omega_h::VERT, "Displacements", SpatialDim,                    Omega_h::Reals(tDisp));
          aMesh.add_tag(Omega_h::VERT, "Pressure",      1 /*output_num_dof_per_node*/, Omega_h::Reals(tPress));

          addElementStateTags(aMesh, aStateDataMap, iStep);
          Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
          tWriter.write(/*time_index*/iStep, /*current_time=*/(Plato::Scalar)iStep, tTags);
        }

    } else
    if(tPhysics == "StructuralDynamics")
    {
    } else 
    if(tPhysics == "Thermal")
    {
        const Plato::Scalar tRestartTime = 0.;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);

        auto nSteps = aState.extent(0);
        for(decltype(nSteps) iStep=0; iStep<nSteps; iStep++){
          auto tSubView = Kokkos::subview(aState, iStep, Kokkos::ALL());
  
          auto tNumVertices = aMesh.nverts();
          Omega_h::Write<Omega_h::Real> tTemp(tNumVertices, "Temperature");
          
          const Plato::OrdinalType tStride = 0;
          Plato::copy<1 /*input_numDof_per_node*/, 1 /*output_numDof_per_node*/>
              (tStride, tNumVertices, tSubView, tTemp);
          
          aMesh.add_tag(Omega_h::VERT, "Temperature", 1 /*output_numDof_per_node*/, Omega_h::Reals(tTemp));

          addElementStateTags(aMesh, aStateDataMap, iStep);
          Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
          tWriter.write(/*time_index*/iStep, /*current_time=*/(Plato::Scalar)iStep, tTags);
        }
    }
}

inline void write(Teuchos::ParameterList & aParamList,
           const std::string & aOutputFilePath,
           const Plato::ScalarMultiVector & aState,
           const Plato::ScalarVector & aControl,
           const Plato::DataMap & aStateDataMap,
           Omega_h::Mesh& aMesh)
{

    auto tNumVertices = aMesh.nverts();
    Omega_h::Write<Omega_h::Real> tControl (tNumVertices, "Control");
    Plato::copy<1, 1>(/*tStride=*/0, tNumVertices, aControl, tControl);
    aMesh.add_tag(Omega_h::VERT, "Control", 1, Omega_h::Reals(tControl));

    const Plato::OrdinalType tSpaceDim = aParamList.get<Plato::OrdinalType>("Spatial Dimension", 3);
    if(tSpaceDim == static_cast<Plato::OrdinalType>(1))
    {
        #ifdef PLATOANALYZE_1D
        Plato::output<1>(aParamList, aOutputFilePath, aState, aStateDataMap, aMesh);
        #else
        throw std::runtime_error("1D physics is not compiled.");
        #endif
    } else
    if(tSpaceDim == static_cast<Plato::OrdinalType>(2))
    {
        #ifdef PLATOANALYZE_2D
        Plato::output<2>(aParamList, aOutputFilePath, aState, aStateDataMap, aMesh);
        #else
        throw std::runtime_error("2D physics is not compiled.");
        #endif
    } else
    if(tSpaceDim == static_cast<Plato::OrdinalType>(3))
    {
        #ifdef PLATOANALYZE_3D
        Plato::output<3>(aParamList, aOutputFilePath, aState, aStateDataMap, aMesh);
        #else
        throw std::runtime_error("3D physics is not compiled.");
        #endif
    }
}


} // namespace Plato

#endif
