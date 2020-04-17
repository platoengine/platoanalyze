/*
 * SurfaceArea.hpp
 *
 *  Created on: Aug 15, 2019
 *      Author: doble
 */

#ifndef SRC_PLATO_SURFACEAREA_HPP_
#define SRC_PLATO_SURFACEAREA_HPP_

#include "BLAS1.hpp"
#include "AbstractScalarFunction.hpp"
#include "ApplyWeighting.hpp"
#include "ImplicitFunctors.hpp"
#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

namespace Plato
{
/******************************************************************************/
template<typename EvaluationType>
class SurfaceArea : public Plato::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    const std::string    name;
    const std::string ss_name = "iside";

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Omega_h::Mesh& mesh; /*!< mesh database */
    Omega_h::MeshSets& meshsets;

  public:
    /**************************************************************************/
    SurfaceArea(Omega_h::Mesh& aMesh,
                Omega_h::MeshSets& aMeshSets,
                Plato::DataMap& aDataMap,
                Teuchos::ParameterList&,
                std::string& aFunctionName) :
                    Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
                    mesh(aMesh),
                    meshsets(aMeshSets)
            /**************************************************************************/
    {

    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> &,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      // get sideset faces
      auto& sidesets = meshsets[Omega_h::SIDE_SET];
      auto ssIter = sidesets.find(this->ss_name);
      auto faceLids = (ssIter->second);
      auto numFaces = faceLids.size();

//      printf("side set name : %s\n", ss_name);
      printf("numFaces : %d\n", numFaces);

      auto face2verts = mesh.ask_verts_of(SpaceDim-1);
      auto cell2verts = mesh.ask_elem_verts();

      auto face2elems = mesh.ask_up(SpaceDim - 1, SpaceDim);
      auto face2elems_map   = face2elems.a2ab;
      auto face2elems_elems = face2elems.ab2b;

      auto nodesPerFace = SpaceDim;
      auto nodesPerCell = SpaceDim+1;

      // create functor for accessing side node coordinates
      Plato::ComputeSurfaceArea<SpaceDim> tComputeCellSurfaceArea;

      // fill the vector with 0s
      Plato::blas1::fill(0.0,  aResult);

      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numFaces), LAMBDA_EXPRESSION(int iFace)
      {
          ConfigScalarType tSurfaceArea;

          // face ordinal
          auto faceOrdinal = faceLids[iFace];

          // allocate local node ordinal array
          int localNodeOrd[SpaceDim];

          // iterate through cells atttached to current face
          for( int localElemOrd = face2elems_map[faceOrdinal]; localElemOrd < face2elems_map[faceOrdinal+1]; ++localElemOrd )
          {
              auto cellOrdinal = face2elems_elems[localElemOrd];

              // iterate through the nodes on the face
              for( int iNode=0; iNode<nodesPerFace; iNode++)
              {
                  // iterate through nodes on the cell
                  for( int jNode=0; jNode<nodesPerCell; jNode++)
                  {
                      // if the current face node matches the node on the cell (store the node ordinal relative to the cell)
                      if( face2verts[faceOrdinal*nodesPerFace+iNode] == cell2verts[cellOrdinal*nodesPerCell + jNode] )
                      {
                          localNodeOrd[iNode] = jNode;
                      }
                  }
              }

              // compute surface area
              tComputeCellSurfaceArea(cellOrdinal, localNodeOrd, aConfig, tSurfaceArea);

              aResult(cellOrdinal) += tSurfaceArea;
          }


      },"SurfaceArea");
    }
};


}



#endif /* SRC_PLATO_SURFACEAREA_HPP_ */
