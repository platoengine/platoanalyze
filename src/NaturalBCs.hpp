#ifndef NATURAL_BC_HPP
#define NATURAL_BC_HPP

#include <sstream>

#include "AnalyzeMacros.hpp"
#include "ImplicitFunctors.hpp"
#include "PlatoStaticsTypes.hpp"

#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

namespace Plato {

  /******************************************************************************/
  /*!
    \brief Class for natural boundary conditions.
  */
    template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs=SpatialDim, Plato::OrdinalType DofsPerNode=NumDofs, Plato::OrdinalType DofOffset=0>
    class NaturalBC
  /******************************************************************************/
  {
    const std::string mName;
    const std::string mSideSetName;
    Omega_h::Vector<NumDofs> mFlux;

  public:

    NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(const std::string & aLoadName, Teuchos::ParameterList &aParam) :
      mName(aLoadName),
      mSideSetName(aParam.get<std::string>("Sides"))
      {
        auto tFlux = aParam.get<Teuchos::Array<Plato::Scalar>>("Vector");
        for(Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
        {
          mFlux(tDof) = tFlux[tDof];
        }
      }

    ~NaturalBC(){}

    /*!
      \brief Get the contribution to the assembled forcing vector.
      @param aMesh Omega_h mesh that contains sidesets.
      @param aMeshSets Omega_h mesh sets that contains sideset information.
      @param aResult Assembled vector to which the boundary terms will be added.

      The boundary terms are integrated on the parameterized surface, \f$\phi(\xi,\psi)\f$, according to:
      \f{eqnarray*}{
          \phi(\xi,\psi)=
            \left\{
             \begin{array}{ccc}
               N_I\left(\xi,\psi\right) x_I &
               N_I\left(\xi,\psi\right) y_I &
               N_I\left(\xi,\psi\right) z_I
             \end{array}
            \right\} \\
          f^{el}_{Ii} = \int_{\partial\Omega_{\xi}} N_I\left(\xi,\psi\right) t_i 
                \left|\left|
                  \frac{\partial\phi}{\partial\xi} \times \frac{\partial\phi}{\partial\psi}
                \right|\right| d\xi d\psi
      \f}
    */

    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void get( Omega_h::Mesh* aMesh,
              const Omega_h::MeshSets& aMeshSets,
              Plato::ScalarMultiVectorT<  StateScalarType>,
              Plato::ScalarMultiVectorT<ControlScalarType>,
              Plato::ScalarArray3DT    < ConfigScalarType>,
              Plato::ScalarMultiVectorT< ResultScalarType>,
              Plato::Scalar aScale) const;

    // ! Get sideset name
    decltype(mSideSetName) const& get_ss_name() const { return mSideSetName; }

    // ! Get the user-specified flux.
    decltype(mFlux) get_value() const { return mFlux; }

  };


  /******************************************************************************/
  /*!
    \brief Owner class that contains a vector of NaturalBC objects.
  */
  template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs=SpatialDim, Plato::OrdinalType DofsPerNode=NumDofs, Plato::OrdinalType DofOffset=0>
  class NaturalBCs
  /******************************************************************************/
  {
  private:
    std::vector<std::shared_ptr<NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>>> mBCs;

  public :

    /*!
      \brief Constructor that parses and creates a vector of NaturalBC objects
      based on the ParameterList.
    */
    NaturalBCs(Teuchos::ParameterList &aParams);

    /*!
      \brief Get the contribution to the assembled forcing vector from the owned boundary conditions.
      @param aMesh Omega_h mesh that contains sidesets.
      @param aMeshSets Omega_h mesh sets that contains sideset information.
      @param aResult Assembled vector to which the boundary terms will be added.
    */
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void get( Omega_h::Mesh* aMesh,
              const Omega_h::MeshSets& aMeshSets,
              Plato::ScalarMultiVectorT<  StateScalarType>,
              Plato::ScalarMultiVectorT<ControlScalarType>,
              Plato::ScalarArray3DT    < ConfigScalarType>,
              Plato::ScalarMultiVectorT< ResultScalarType>,
              Plato::Scalar scale = 1.0) const;
  };

  /**************************************************************************/
  template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
  template<typename StateScalarType,
           typename ControlScalarType,
           typename ConfigScalarType,
           typename ResultScalarType>
  void NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get( Omega_h::Mesh* aMesh,
                                               const Omega_h::MeshSets& aMeshSets,
                                               Plato::ScalarMultiVectorT<  StateScalarType> aState,
                                               Plato::ScalarMultiVectorT<ControlScalarType> aControl,
                                               Plato::ScalarArray3DT    < ConfigScalarType> aConfig,
                                               Plato::ScalarMultiVectorT< ResultScalarType> aResult,
                                               Plato::Scalar aScale) const
  /**************************************************************************/
  {
    // get sideset faces
    auto& tSideSets = aMeshSets[Omega_h::SIDE_SET];
    auto tSideSetMapIterator = tSideSets.find(this->mSideSetName);
    if(tSideSetMapIterator == tSideSets.end())
    {
        std::ostringstream tMsg;
        tMsg << "COULD NOT FIND SIDE SET WITH NAME = '" << this->mSideSetName.c_str()
            << "'.  SIDE SET IS NOT DEFINED IN THE INPUT MESH FILE, I.E. EXODUS FILE.\n";
        THROWERR(tMsg.str());
    }
    auto tFaceLids = (tSideSetMapIterator->second);
    auto tNumFaces = tFaceLids.size();


    // get mesh vertices
    auto tFace2Verts = aMesh->ask_verts_of(SpatialDim-1);
    auto tCell2Verts = aMesh->ask_elem_verts();

    auto tFace2eElems = aMesh->ask_up(SpatialDim - 1, SpatialDim);
    auto tFace2Elems_map   = tFace2eElems.a2ab;
    auto tFace2Elems_elems = tFace2eElems.ab2b;

    auto tNodesPerFace = SpatialDim;
    auto tNodesPerCell = SpatialDim+1;

    Plato::ScalarArray3DT<ConfigScalarType> tJacobian("jacobian", tNumFaces, SpatialDim-1, SpatialDim);

    auto tFlux = mFlux;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceIndex)
    {

      auto tFaceOrdinal = tFaceLids[aFaceIndex];

      // for each element that the face is connected to: (either 1 or 2)
      for( Plato::OrdinalType tLocalElemOrd = tFace2Elems_map[tFaceOrdinal]; tLocalElemOrd < tFace2Elems_map[tFaceOrdinal+1]; ++tLocalElemOrd )
      {

        // create a map from face local node index to elem local node index
        Plato::OrdinalType tLocalNodeOrd[SpatialDim];
        auto tCellOrdinal = tFace2Elems_elems[tLocalElemOrd];
        for( Plato::OrdinalType tNodeI=0; tNodeI<tNodesPerFace; tNodeI++)
        {
          for( Plato::OrdinalType tNodeJ=0; tNodeJ<tNodesPerCell; tNodeJ++)
          {
            if( tFace2Verts[tFaceOrdinal*tNodesPerFace+tNodeI] == tCell2Verts[tCellOrdinal*tNodesPerCell + tNodeJ] )
            {
              tLocalNodeOrd[tNodeI] = tNodeJ;
            }
          }
        }

        // compute jacobian from aConfig
        for( Plato::OrdinalType tNode=0; tNode<SpatialDim-1; tNode++)
        {
          for( Plato::OrdinalType tDim=0; tDim<SpatialDim; tDim++)
          {
            tJacobian(aFaceIndex,tNode,tDim) =
                    aConfig(tCellOrdinal, tLocalNodeOrd[tNode], tDim) - aConfig(tCellOrdinal, tLocalNodeOrd[SpatialDim-1], tDim);
          }
        }

        ConfigScalarType tWeight(0.0);
        if(SpatialDim==1)
        {
          tWeight = aScale;
        } else
        if(SpatialDim==2)
        {
          tWeight = aScale/2.0*sqrt(tJacobian(aFaceIndex,0,0)*tJacobian(aFaceIndex,0,0)+tJacobian(aFaceIndex,0,1)*tJacobian(aFaceIndex,0,1));
        } else
        if(SpatialDim==3)
        {
          auto tJ23 = tJacobian(aFaceIndex,0,1)*tJacobian(aFaceIndex,1,2)-tJacobian(aFaceIndex,0,2)*tJacobian(aFaceIndex,1,1);
          auto tJ31 = tJacobian(aFaceIndex,0,2)*tJacobian(aFaceIndex,1,0)-tJacobian(aFaceIndex,0,0)*tJacobian(aFaceIndex,1,2);
          auto tJ12 = tJacobian(aFaceIndex,0,0)*tJacobian(aFaceIndex,1,1)-tJacobian(aFaceIndex,0,1)*tJacobian(aFaceIndex,1,0);
          tWeight = aScale/6.0*sqrt(tJ23*tJ23+tJ31*tJ31+tJ12*tJ12);
        }

        // project into aResult workset
        for( Plato::OrdinalType tNode=0; tNode<tNodesPerFace; tNode++)
        {
          for( Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
          {
            auto tCellDofOrdinal = tLocalNodeOrd[tNode] * DofsPerNode + tDof + DofOffset;
            aResult(tCellOrdinal,tCellDofOrdinal) += tWeight*tFlux[tDof];
          }
        }
      }

    });
  }

  /****************************************************************************/
  template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
  NaturalBCs<SpatialDim,NumDofs,DofsPerNode,DofOffset>::NaturalBCs(Teuchos::ParameterList &aParams) : mBCs()
  /****************************************************************************/
  {
    for (Teuchos::ParameterList::ConstIterator tItr = aParams.begin(); tItr != aParams.end(); ++tItr) {
      const Teuchos::ParameterEntry &tEntry = aParams.entry(tItr);
      const std::string             &tName  = aParams.name(tItr);

      TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(),
         std::logic_error,
         "Parameter in Boundary Conditions block not valid.  Expect lists only.");

      Teuchos::ParameterList& tSubList = aParams.sublist(tName);
      const std::string tType = tSubList.get<std::string>("Type");
      std::shared_ptr<NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>> tBC;
      if ("Uniform" == tType) {
        bool tBC_Values = tSubList.isType<Teuchos::Array<Plato::Scalar>>("Values");
        bool tBC_Value  = tSubList.isType<Plato::Scalar>("Value");
        if ( tBC_Values && tBC_Value ) {
          TEUCHOS_TEST_FOR_EXCEPTION(true,
             std::logic_error,
             " Natural Boundary Condition: provide EITHER 'Values' OR 'Value' Parameter.");
        } else
        if ( tBC_Values ) {
          auto tValues = tSubList.get<Teuchos::Array<Plato::Scalar>>("Values");
          tSubList.set("Vector", tValues);
        } else
        if ( tBC_Value ) {
          Teuchos::Array<Plato::Scalar> tFluxVector(NumDofs, 0.0);
          auto tValue = tSubList.get<Plato::Scalar>("Value");
          tFluxVector[0] = tValue;
          tSubList.set("Vector", tFluxVector);
        } else {
          TEUCHOS_TEST_FOR_EXCEPTION(true,
             std::logic_error,
             " Natural Boundary Condition: provide either 'Values' or 'Value' Parameter.");
        }
        tBC.reset(new NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(tName, tSubList));
      }
      else if ("Uniform Component" == tType){
        Teuchos::Array<Plato::Scalar> tFluxVector(NumDofs, 0.0);
        auto tFluxComponent = tSubList.get<std::string>("Component");
        auto tValue = tSubList.get<Plato::Scalar>("Value");
        if( (tFluxComponent == "x" || tFluxComponent == "X") ) tFluxVector[0] = tValue;
        else
        if( (tFluxComponent == "y" || tFluxComponent == "Y") && DofsPerNode > 1 ) tFluxVector[1] = tValue;
        else
        if( (tFluxComponent == "z" || tFluxComponent == "Z") && DofsPerNode > 2 ) tFluxVector[2] = tValue;
        tSubList.set("Vector", tFluxVector);
        tBC.reset(new NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(tName, tSubList));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true,
           std::logic_error,
           " Natural Boundary Condition type invalid");
      }
      mBCs.push_back(tBC);
    }
  }

  /**************************************************************************/
  /*!
    \brief Add the boundary load to the result workset
  */
  template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
  template<typename StateScalarType,
           typename ControlScalarType,
           typename ConfigScalarType,
           typename ResultScalarType>
  void NaturalBCs<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get(Omega_h::Mesh* aMesh,
       const Omega_h::MeshSets& aMeshSets,
       Plato::ScalarMultiVectorT<  StateScalarType> aState,
       Plato::ScalarMultiVectorT<ControlScalarType> aControl,
       Plato::ScalarArray3DT    < ConfigScalarType> aConfig,
       Plato::ScalarMultiVectorT< ResultScalarType> aResult,
       Plato::Scalar aScale) const
  /**************************************************************************/
  {
    for (const auto &bc : mBCs){
      bc->get(aMesh, aMeshSets, aState, aControl, aConfig, aResult, aScale);
    }
  }


} // end namespace Plato 

#endif

