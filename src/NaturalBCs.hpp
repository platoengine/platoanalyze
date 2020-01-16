#ifndef NATURAL_BC_HPP
#define NATURAL_BC_HPP

#include <sstream>

#include "ImplicitFunctors.hpp"

#include "PlatoStaticsTypes.hpp"
#include "AnalyzeMacros.hpp"

#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

<<<<<<< HEAD:src/NaturalBCs.hpp
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
              Plato::Scalar scale) const;
=======

/******************************************************************************//**
 * \brief Natural (i.e. Neumann) boundary condition class
 *
 * \tparam SpatialDim  number of spatial dimensions
 * \tparam NumDofs     number of force degrees of freedom
 * \tparam DofsPerNode degrees of freedom per node
 * \tparam DofOffset   degree of freedom offset
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim,
         Plato::OrdinalType NumDofs = SpatialDim,
         Plato::OrdinalType DofsPerNode = NumDofs,
         Plato::OrdinalType DofOffset = 0>
class NaturalBC
{
// private access member data
private:
    const std::string mName;         /*!< Neumann BC name */
    const std::string mSideSetName;  /*!< Neumann BC side set name */
    Omega_h::Vector<NumDofs> mValues;  /*!< Neumann BC values */

// public access functions
public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aName        Neumann BC name
     * \param [in] aParamInputs input parameters provided in XML file
    *******************************************************************************/
    NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>(const std::string &aName, Teuchos::ParameterList &aParamInputs) :
            mName(aName),
            mSideSetName(aParamInputs.get < std::string > ("Sides"))
    {
        auto tFlux = aParamInputs.get < Teuchos::Array < Plato::Scalar >> ("Vector");
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofs; tDofIndex++)
        {
            mValues(tDofIndex) = tFlux[tDofIndex];
        }
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    ~NaturalBC()
    {
    }

    /***************************************************************************//**
     * \brief Get the contribution to the assembled forcing vector.
     *
     * \tparam StateScalarType   POD type of Kokkos::View
     * \tparam ControlScalarType POD type of Kokkos::View
     * \tparam ResultScalarType  POD type of Kokkos::View
     *
     * \param aMesh         Omega_h mesh that contains sidesets.
     * \param aMeshSets     Omega_h mesh sets that contains sideset information.
     * \param aGlobalStates global state variables workset.
     * \param aControls     control variables workset.
     * \param aResult       assembled vector to which the boundary terms will be added.
     * \param aScale        scalar multiplier
     *
     * The boundary terms are integrated on the parameterized surface, \f$\phi(\xi,\psi)\f$, according to:
     * \f{eqnarray*}{
     * \phi(\xi,\psi)=
     * \left\{
     * \begin{array}{ccc}
     * N_I\left(\xi,\psi\right) x_I &
     * N_I\left(\xi,\psi\right) y_I &
     * N_I\left(\xi,\psi\right) z_I
     * \end{array}
     * \right\} \\
     *      f^{el}_{Ii} = \int_{\partial\Omega_{\xi}} N_I\left(\xi,\psi\right) t_i
     * \left|\left|
     * \frac{\partial\phi}{\partial\xi} \times \frac{\partial\phi}{\partial\psi}
     * \right|\right| d\xi d\psi
     * \f}
     *
    *******************************************************************************/
    template<typename StateScalarType, typename ControlScalarType, typename ResultScalarType>
    void get(Omega_h::Mesh* aMesh,
             const Omega_h::MeshSets& aMeshSets,
             Plato::ScalarMultiVectorT<StateScalarType> aGlobalStates,
             Plato::ScalarMultiVectorT<ControlScalarType> aControls,
             Plato::ScalarMultiVectorT<ResultScalarType> aResult,
             Plato::Scalar aScale) const;
>>>>>>> f7210115e204d8d5ae0b5d58ef095e9320f4fe71:src/plato/NaturalBCs.hpp

    // ! Get sideset name
    decltype(mSideSetName) const& get_ss_name() const
    {
        return mSideSetName;
    }

    // ! Get the user-specified flux.
    decltype(mValues) get_value() const
    {
        return mValues;
    }

};

/******************************************************************************/
/*!
 \brief Owner class that contains a vector of NaturalBC objects.
 */
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs = SpatialDim, Plato::OrdinalType DofsPerNode = NumDofs, Plato::OrdinalType DofOffset = 0>
class NaturalBCs
/******************************************************************************/
{
private:
    std::vector<std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>> BCs;

public:

    /*!
     \brief Constructor that parses and creates a vector of NaturalBC objects
     based on the ParameterList.
     */
    NaturalBCs(Teuchos::ParameterList &aParams);

    /*!
<<<<<<< HEAD:src/NaturalBCs.hpp
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
  template<int SpatialDim, int NumDofs, int DofsPerNode, int DofOffset>
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
                                               Plato::Scalar scale) const
  /**************************************************************************/
  {
    // get sideset faces
    auto& sidesets = aMeshSets[Omega_h::SIDE_SET];
    auto ssIter = sidesets.find(this->ss_name);
    auto faceLids = (ssIter->second);
    auto numFaces = faceLids.size();
=======
     \brief Get the contribution to the assembled forcing vector from the owned boundary conditions.
     \param aMesh Omega_h mesh that contains sidesets.
     \param aMeshSets Omega_h mesh sets that contains sideset information.
     \param aResult Assembled vector to which the boundary terms will be added.
     */
    template<typename StateScalarType, typename ControlScalarType, typename ResultScalarType>
    void get(Omega_h::Mesh* aMesh,
             const Omega_h::MeshSets& aMeshSets,
             Plato::ScalarMultiVectorT<StateScalarType> aGlobalStates,
             Plato::ScalarMultiVectorT<ControlScalarType> aControls,
             Plato::ScalarMultiVectorT<ResultScalarType> aResult,
             Plato::Scalar aScale = 1.0) const;
};
>>>>>>> f7210115e204d8d5ae0b5d58ef095e9320f4fe71:src/plato/NaturalBCs.hpp

/**************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType, typename ControlScalarType, typename ResultScalarType>
void NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>::get(Omega_h::Mesh* aMesh,
                                                                 const Omega_h::MeshSets& aMeshSets,
                                                                 Plato::ScalarMultiVectorT<StateScalarType>,
                                                                 Plato::ScalarMultiVectorT<ControlScalarType>,
                                                                 Plato::ScalarMultiVectorT<ResultScalarType> aResult,
                                                                 Plato::Scalar scale) const
                                                                 /**************************************************************************/
{
    // get sideset faces
    auto& tSideSets = aMeshSets[Omega_h::SIDE_SET];
    auto tSideSetIter = tSideSets.find(this->mSideSetName);
    if(tSideSetIter == tSideSets.end())
    {
        std::ostringstream tMsg;
        tMsg << "Could not find Side Set with name = '" << mSideSetName.c_str()
                << "'. Side Set is not defined in input geometry/mesh file.\n";
        THROWERR(tMsg.str())
    }
    auto tFaceLids = (tSideSetIter->second);
    auto tNumFaces = tFaceLids.size();

    // get mesh vertices
    auto tFace2verts = aMesh->ask_verts_of(SpatialDim - 1);
    auto tCell2verts = aMesh->ask_elem_verts();

    auto tFace2elems = aMesh->ask_up(SpatialDim - 1, SpatialDim);
    auto tFace2elems_map = tFace2elems.a2ab;
    auto tFace2elems_elems = tFace2elems.ab2b;

    auto tNodesPerFace = SpatialDim;
    auto tNodesPerCell = SpatialDim + 1;

<<<<<<< HEAD:src/NaturalBCs.hpp
    Plato::ScalarArray3DT<ConfigScalarType> tJacobian("jacobian", numFaces, SpatialDim-1, SpatialDim);

    auto flux = mFlux;
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numFaces), LAMBDA_EXPRESSION(int iFace)
    {
=======
    // create functor for accessing side node coordinates
    Plato::SideNodeCoordinate<SpatialDim> tSideNodeCoordinate(aMesh);
>>>>>>> f7210115e204d8d5ae0b5d58ef095e9320f4fe71:src/plato/NaturalBCs.hpp

    auto tValues = mValues;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumFaces), LAMBDA_EXPRESSION(Plato::OrdinalType iFace)
    {
        auto tFaceOrdinal = tFaceLids[iFace];

<<<<<<< HEAD:src/NaturalBCs.hpp
      // for each element that the face is connected to: (either 1 or 2)
      for( int localElemOrd = face2elems_map[faceOrdinal]; localElemOrd < face2elems_map[faceOrdinal+1]; ++localElemOrd ){

        // create a map from face local node index to elem local node index
        int localNodeOrd[SpatialDim];
        auto cellOrdinal = face2elems_elems[localElemOrd];
        for( int iNode=0; iNode<nodesPerFace; iNode++){
          for( int jNode=0; jNode<nodesPerCell; jNode++){
            if( face2verts[faceOrdinal*nodesPerFace+iNode] == cell2verts[cellOrdinal*nodesPerCell + jNode] ) localNodeOrd[iNode] = jNode;
          }
        }

        // compute jacobian from aConfig
        for( int iNode=0; iNode<SpatialDim-1; iNode++){
          for( int iDim=0; iDim<SpatialDim; iDim++){
            tJacobian(iFace,iNode,iDim) = aConfig(cellOrdinal, localNodeOrd[iNode], iDim)
                                        - aConfig(cellOrdinal, localNodeOrd[SpatialDim-1], iDim);
          }
        }
        ConfigScalarType weight(0.0);
        if(SpatialDim==1){
          weight=scale;
        } else
        if(SpatialDim==2){
          weight = scale/2.0*sqrt(tJacobian(iFace,0,0)*tJacobian(iFace,0,0)+tJacobian(iFace,0,1)*tJacobian(iFace,0,1));
        } else
        if(SpatialDim==3){
          auto a1 = tJacobian(iFace,0,1)*tJacobian(iFace,1,2)-tJacobian(iFace,0,2)*tJacobian(iFace,1,1);
          auto a2 = tJacobian(iFace,0,2)*tJacobian(iFace,1,0)-tJacobian(iFace,0,0)*tJacobian(iFace,1,2);
          auto a3 = tJacobian(iFace,0,0)*tJacobian(iFace,1,1)-tJacobian(iFace,0,1)*tJacobian(iFace,1,0);
          weight = scale/6.0*sqrt(a1*a1+a2*a2+a3*a3);
        }

        // project into aResult workset
        for( int iNode=0; iNode<nodesPerFace; iNode++){
          for( int iDof=0; iDof<NumDofs; iDof++){
            auto cellDofOrdinal = localNodeOrd[iNode] * DofsPerNode + iDof + DofOffset;
            aResult(cellOrdinal,cellDofOrdinal) += weight*flux[iDof];
          }
        }
      }

=======
        // integrate
        //
        Omega_h::Matrix<SpatialDim, SpatialDim-1> tJacobian;
        for (Plato::OrdinalType d1=0; d1<SpatialDim-1; d1++)
        {
            for (Plato::OrdinalType d2=0; d2<SpatialDim; d2++)
            {
                tJacobian[d1][d2] = tSideNodeCoordinate(tFaceOrdinal,d1,d2) - tSideNodeCoordinate(tFaceOrdinal,SpatialDim-1,d2);
            }
        }

        Plato::Scalar tWeight(0.0);
        if(SpatialDim==1)
        {
            tWeight=scale;
        }
        else
            if(SpatialDim==2)
            {
                tWeight = scale/2.0*sqrt(tJacobian[0][0]*tJacobian[0][0]+tJacobian[0][1]*tJacobian[0][1]);
            }
            else
                if(SpatialDim==3)
                {
                    auto a1 = tJacobian[0][1]*tJacobian[1][2]-tJacobian[0][2]*tJacobian[1][1];
                    auto a2 = tJacobian[0][2]*tJacobian[1][0]-tJacobian[0][0]*tJacobian[1][2];
                    auto a3 = tJacobian[0][0]*tJacobian[1][1]-tJacobian[0][1]*tJacobian[1][0];
                    tWeight = scale/6.0*sqrt(a1*a1+a2*a2+a3*a3);
                }

        Plato::OrdinalType tLocalNodeOrd[SpatialDim];
        for( Plato::OrdinalType tLocalElemOrd = tFace2elems_map[tFaceOrdinal];
                tLocalElemOrd < tFace2elems_map[tFaceOrdinal+1]; ++tLocalElemOrd )
        {
            auto tCellOrdinal = tFace2elems_elems[tLocalElemOrd];
            for( Plato::OrdinalType iNode=0; iNode<tNodesPerFace; iNode++)
            {
                for( Plato::OrdinalType jNode=0; jNode<tNodesPerCell; jNode++)
                {
                    if( tFace2verts[tFaceOrdinal*tNodesPerFace+iNode] == tCell2verts[tCellOrdinal*tNodesPerCell + jNode] ) tLocalNodeOrd[iNode] = jNode;
                }
            }
            for( Plato::OrdinalType iNode=0; iNode<tNodesPerFace; iNode++)
            {
                for( Plato::OrdinalType iDof=0; iDof<NumDofs; iDof++)
                {
                    auto tCellDofOrdinal = tLocalNodeOrd[iNode] * DofsPerNode + iDof + DofOffset;
                    aResult(tCellOrdinal,tCellDofOrdinal) += tWeight*tValues[iDof];
                }
            }
        }
>>>>>>> f7210115e204d8d5ae0b5d58ef095e9320f4fe71:src/plato/NaturalBCs.hpp
    });
}

/****************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
NaturalBCs<SpatialDim, NumDofs, DofsPerNode, DofOffset>::NaturalBCs(Teuchos::ParameterList &aParams) :
        BCs()
/****************************************************************************/
{
    for(Teuchos::ParameterList::ConstIterator i = aParams.begin(); i != aParams.end(); ++i)
    {
        const Teuchos::ParameterEntry &entry = aParams.entry(i);
        const std::string &name = aParams.name(i);

        TEUCHOS_TEST_FOR_EXCEPTION(!entry.isList(),
                                   std::logic_error,
                                   "Parameter in Boundary Conditions block not valid.  Expect lists only.");

        Teuchos::ParameterList& sublist = aParams.sublist(name);
        const std::string type = sublist.get < std::string > ("Type");
        std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>> bc;
        if("Uniform" == type)
        {
            bool b_Values = sublist.isType < Teuchos::Array < Plato::Scalar >> ("Values");
            bool b_Value = sublist.isType < Plato::Scalar > ("Value");
            if(b_Values && b_Value)
            {
                TEUCHOS_TEST_FOR_EXCEPTION(true,
                                           std::logic_error,
                                           " Natural Boundary Condition: provide EITHER 'Values' OR 'Value' Parameter.");
            }
            else if(b_Values)
            {
                auto values = sublist.get < Teuchos::Array < Plato::Scalar >> ("Values");
                sublist.set("Vector", values);
            }
            else if(b_Value)
            {
                Teuchos::Array<Plato::Scalar> fluxVector(NumDofs, 0.0);
                auto value = sublist.get < Plato::Scalar > ("Value");
                fluxVector[0] = value;
                sublist.set("Vector", fluxVector);
            }
            else
            {
                TEUCHOS_TEST_FOR_EXCEPTION(true,
                                           std::logic_error,
                                           " Natural Boundary Condition: provide either 'Values' or 'Value' Parameter.");
            }
            bc.reset(new NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>(name, sublist));
        }
        else if("Uniform Component" == type)
        {
            Teuchos::Array<Plato::Scalar> fluxVector(NumDofs, 0.0);
            auto fluxComponent = sublist.get < std::string > ("Component");
            auto value = sublist.get < Plato::Scalar > ("Value");
            if((fluxComponent == "x" || fluxComponent == "X"))
                fluxVector[0] = value;
            else if((fluxComponent == "y" || fluxComponent == "Y") && DofsPerNode > 1)
                fluxVector[1] = value;
            else if((fluxComponent == "z" || fluxComponent == "Z") && DofsPerNode > 2)
                fluxVector[2] = value;
            sublist.set("Vector", fluxVector);
            bc.reset(new NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>(name, sublist));
        }
        else
<<<<<<< HEAD:src/NaturalBCs.hpp
        if( (fluxComponent == "z" || fluxComponent == "Z") && DofsPerNode > 2 ) fluxVector[2] = value;
        sublist.set("Vector", fluxVector);
        bc.reset(new NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(name, sublist));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, 
           std::logic_error,
           " Natural Boundary Condition type invalid");
      }
      BCs.push_back(bc);
    }
  }

  /**************************************************************************/
  /*!
    \brief Add the boundary load to the result workset
  */
  template<int SpatialDim, int NumDofs, int DofsPerNode, int DofOffset>
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
    for (const auto &bc : BCs){
      bc->get(aMesh, aMeshSets, aState, aControl, aConfig, aResult, aScale);
=======
        {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, " Natural Boundary Condition type invalid");
        }
        BCs.push_back(bc);
>>>>>>> f7210115e204d8d5ae0b5d58ef095e9320f4fe71:src/plato/NaturalBCs.hpp
    }
}

/**************************************************************************/
/*!
 \brief Add the boundary load to the result workset
 */
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType, typename ControlScalarType, typename ResultScalarType>
void NaturalBCs<SpatialDim, NumDofs, DofsPerNode, DofOffset>::get(Omega_h::Mesh* aMesh,
                                                                  const Omega_h::MeshSets& aMeshSets,
                                                                  Kokkos::View<StateScalarType**, Kokkos::LayoutRight,
                                                                          Plato::MemSpace> state,
                                                                  Kokkos::View<ControlScalarType**, Kokkos::LayoutRight,
                                                                          Plato::MemSpace> control,
                                                                  Kokkos::View<ResultScalarType**, Kokkos::LayoutRight,
                                                                          Plato::MemSpace> result,
                                                                  Plato::Scalar scale) const
                                                                  /**************************************************************************/
{
    for(const auto &bc : BCs)
    {
        bc->get(aMesh, aMeshSets, state, control, result, scale);
    }
}

} // end namespace Plato 

#endif

