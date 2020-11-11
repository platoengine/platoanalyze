#ifndef PLATO_IMPLICIT_FUNCTORS_HPP
#define PLATO_IMPLICIT_FUNCTORS_HPP

#include <Omega_h_mesh.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_vector.hpp>

#include <Teuchos_RCPDecl.hpp>
#include "SimplexMechanics.hpp"

namespace Plato
{

/******************************************************************************//**
* \brief functor that provides mesh-local node ordinal
* \param [in] aMesh Omega_h mesh
**********************************************************************************/
template<Plato::OrdinalType SpaceDim>
class NodeOrdinal
{
  public:
    const Omega_h::LOs mCells2nodes;

  public:
    NodeOrdinal(
      Omega_h::Mesh* aMesh ) :
      mCells2nodes(aMesh->ask_elem_verts()) {}

    /******************************************************************************//**
    * \brief Returns mesh-local node ordinal
    * \param [in] aCellOrdinal mesh-local element ordinal
    * \param [in] aNodeOrdinal elem-local node ordinal
    **********************************************************************************/
    DEVICE_TYPE inline
    Plato::OrdinalType
    operator()(
        Plato::OrdinalType aCellOrdinal,
        Plato::OrdinalType aNodeOrdinal
    ) const
    {
        return mCells2nodes[aCellOrdinal*(SpaceDim+1) + aNodeOrdinal];
    }
};
/******************************************************************************/


/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType DofsPerNode=1>
class VectorEntryOrdinal
{
  public:
    const Omega_h::LOs mCells2nodes;

  public:
    VectorEntryOrdinal(
      Omega_h::Mesh* mesh ) :
      mCells2nodes(mesh->ask_elem_verts()) {}

    DEVICE_TYPE inline Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType nodeOrdinal, Plato::OrdinalType dofOrdinal=0) const
    {
        Plato::OrdinalType vertexNumber = mCells2nodes[cellOrdinal*(SpaceDim+1) + nodeOrdinal];
        return vertexNumber * DofsPerNode + dofOrdinal;
    }
};
/******************************************************************************/


/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class NodeCoordinate
{
  private:
    const Omega_h::LOs mCells2nodes;
    const Omega_h::Reals mCoords;

  public:
    NodeCoordinate(Omega_h::Mesh* aMesh) :
      mCells2nodes(aMesh->ask_elem_verts()),
      mCoords(aMesh->coords()) { }

    DEVICE_TYPE
    inline
    Plato::Scalar
    operator()(Plato::OrdinalType aCellOrdinal, Plato::OrdinalType aNodeOrdinal, Plato::OrdinalType aDimOrdinal) const
    {
        const Plato::OrdinalType tVertexNumber = mCells2nodes[aCellOrdinal*(SpaceDim+1) + aNodeOrdinal];
        const Plato::Scalar tCoord = mCoords[tVertexNumber * SpaceDim + aDimOrdinal];
        return tCoord;
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class JacobianDet
{
  private:
    const NodeCoordinate<SpaceDim> mNodeCoordinate;

  public:
    JacobianDet( Omega_h::Mesh* mesh ) :
      mNodeCoordinate(mesh) {}

    DEVICE_TYPE
    inline
    Plato::Scalar
    operator()(Plato::OrdinalType cellOrdinal) const {
      Omega_h::Matrix<SpaceDim, SpaceDim> jacobian;

      for (Plato::OrdinalType d1=0; d1<SpaceDim; d1++) {
        for (Plato::OrdinalType d2=0; d2<SpaceDim; d2++) {
          jacobian[d1][d2] = mNodeCoordinate(cellOrdinal,d2,d1) - mNodeCoordinate(cellOrdinal,SpaceDim,d1);
        }
      }
      return Omega_h::determinant(jacobian);
    }
};
/******************************************************************************/


/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class SideNodeCoordinate
{
  private:
    const Omega_h::LOs mSides2nodes;
    const Omega_h::Reals mCoords;

  public:
    SideNodeCoordinate(
      Omega_h::Mesh* mesh ) :
      mSides2nodes(mesh->ask_verts_of(SpaceDim-1)),
      mCoords(mesh->coords()) { }

    DEVICE_TYPE
    inline
    Plato::Scalar
    operator()(Plato::OrdinalType sideOrdinal, Plato::OrdinalType nodeOrdinal, Plato::OrdinalType dofOrdinal) const
    {
        Plato::OrdinalType vertexNumber = mSides2nodes[sideOrdinal*SpaceDim + nodeOrdinal];
        Plato::Scalar coord = mCoords[vertexNumber*SpaceDim + dofOrdinal];
        return coord;
    }
};
/******************************************************************************/

/******************************************************************************/
/*! InertialForces Functor.
*
*   Evaluates cell inertial forces.
*/
/******************************************************************************/
class ComputeProjectionWorkset
{
public:
    /******************************************************************************/
    template<typename GaussPointScalarType, typename ProjectedScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                                       const Plato::ScalarVectorT<Plato::Scalar> & tBasisFunctions,
                                       const Plato::ScalarMultiVectorT<GaussPointScalarType> & aStateValues,
                                       const Plato::ScalarMultiVectorT<ProjectedScalarType> & aResult,
                                             Plato::Scalar scale = 1.0 ) const
    /******************************************************************************/
    {
        const Plato::OrdinalType tNumNodesPerCell = tBasisFunctions.size();
        const Plato::OrdinalType tNumDofsPerNode = aStateValues.extent(1);
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerNode; tDofIndex++)
            {
                Plato::OrdinalType tMyDofIndex = (tNumDofsPerNode * tNodeIndex) + tDofIndex;
                aResult(aCellOrdinal, tMyDofIndex) += scale * tBasisFunctions(tNodeIndex)
                        * aStateValues(aCellOrdinal, tDofIndex) * aCellVolume(aCellOrdinal);
            }
        }
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeGradientWorkset
{
  private:
    Omega_h::Matrix<SpaceDim, SpaceDim> jacobian, jacobianInverse;

  public:
    template<typename ScalarType>
    DEVICE_TYPE inline void
    operator()(Plato::OrdinalType aCellOrdinal,
               Plato::ScalarArray3DT<ScalarType> aGradients,
               Plato::ScalarArray3DT<ScalarType> aConfig,
               Plato::ScalarVectorT<ScalarType> aCellVolume) const;
};

  template<>
  template<typename ScalarType>
  DEVICE_TYPE inline void
  ComputeGradientWorkset<3>::operator()(Plato::OrdinalType aCellOrdinal,
                                        Plato::ScalarArray3DT<ScalarType> aGradients,
                                        Plato::ScalarArray3DT<ScalarType> aConfig,
                                        Plato::ScalarVectorT<ScalarType> aCellVolume) const
  {
    ScalarType j11=aConfig(aCellOrdinal,0,0)-aConfig(aCellOrdinal,3,0);
    ScalarType j12=aConfig(aCellOrdinal,1,0)-aConfig(aCellOrdinal,3,0);
    ScalarType j13=aConfig(aCellOrdinal,2,0)-aConfig(aCellOrdinal,3,0);

    ScalarType j21=aConfig(aCellOrdinal,0,1)-aConfig(aCellOrdinal,3,1);
    ScalarType j22=aConfig(aCellOrdinal,1,1)-aConfig(aCellOrdinal,3,1);
    ScalarType j23=aConfig(aCellOrdinal,2,1)-aConfig(aCellOrdinal,3,1);

    ScalarType j31=aConfig(aCellOrdinal,0,2)-aConfig(aCellOrdinal,3,2);
    ScalarType j32=aConfig(aCellOrdinal,1,2)-aConfig(aCellOrdinal,3,2);
    ScalarType j33=aConfig(aCellOrdinal,2,2)-aConfig(aCellOrdinal,3,2);

    ScalarType detj = j11*j22*j33+j12*j23*j31+j13*j21*j32
                     -j11*j23*j32-j12*j21*j33-j13*j22*j31;

    ScalarType i11 = (j22*j33-j23*j32)/detj;
    ScalarType i12 = (j13*j32-j12*j33)/detj;
    ScalarType i13 = (j12*j23-j13*j22)/detj;

    ScalarType i21 = (j23*j31-j21*j33)/detj;
    ScalarType i22 = (j11*j33-j13*j31)/detj;
    ScalarType i23 = (j13*j21-j11*j23)/detj;

    ScalarType i31 = (j21*j32-j22*j31)/detj;
    ScalarType i32 = (j12*j31-j11*j32)/detj;
    ScalarType i33 = (j11*j22-j12*j21)/detj;

    aCellVolume(aCellOrdinal) = fabs(detj);

    aGradients(aCellOrdinal,0,0) = i11;
    aGradients(aCellOrdinal,0,1) = i12;
    aGradients(aCellOrdinal,0,2) = i13;

    aGradients(aCellOrdinal,1,0) = i21;
    aGradients(aCellOrdinal,1,1) = i22;
    aGradients(aCellOrdinal,1,2) = i23;

    aGradients(aCellOrdinal,2,0) = i31;
    aGradients(aCellOrdinal,2,1) = i32;
    aGradients(aCellOrdinal,2,2) = i33;

    aGradients(aCellOrdinal,3,0) = -(i11+i21+i31);
    aGradients(aCellOrdinal,3,1) = -(i12+i22+i32);
    aGradients(aCellOrdinal,3,2) = -(i13+i23+i33);
  }

  template<>
  template<typename ScalarType>
  DEVICE_TYPE inline void
  ComputeGradientWorkset<2>::operator()(Plato::OrdinalType cellOrdinal,
                                        Plato::ScalarArray3DT<ScalarType> gradients,
                                        Plato::ScalarArray3DT<ScalarType> config,
                                        Plato::ScalarVectorT<ScalarType> cellVolume) const
  {
    ScalarType j11=config(cellOrdinal,0,0)-config(cellOrdinal,2,0);
    ScalarType j12=config(cellOrdinal,1,0)-config(cellOrdinal,2,0);

    ScalarType j21=config(cellOrdinal,0,1)-config(cellOrdinal,2,1);
    ScalarType j22=config(cellOrdinal,1,1)-config(cellOrdinal,2,1);

    ScalarType detj = j11*j22-j12*j21;

    ScalarType i11 = j22/detj;
    ScalarType i12 =-j12/detj;

    ScalarType i21 =-j21/detj;
    ScalarType i22 = j11/detj;

    cellVolume(cellOrdinal) = fabs(detj);

    gradients(cellOrdinal,0,0) = i11;
    gradients(cellOrdinal,0,1) = i12;

    gradients(cellOrdinal,1,0) = i21;
    gradients(cellOrdinal,1,1) = i22;

    gradients(cellOrdinal,2,0) = -(i11+i21);
    gradients(cellOrdinal,2,1) = -(i12+i22);
  }

  template<>
  template<typename ScalarType>
  DEVICE_TYPE inline void
  ComputeGradientWorkset<1>::operator()(Plato::OrdinalType cellOrdinal,
                                        Plato::ScalarArray3DT<ScalarType> gradients,
                                        Plato::ScalarArray3DT<ScalarType> config,
                                        Plato::ScalarVectorT<ScalarType> cellVolume) const
  {
    ScalarType j11=config(cellOrdinal,0,0)-config(cellOrdinal,1,0);

    ScalarType detj = j11;

    ScalarType i11 = 1.0/detj;

    cellVolume(cellOrdinal) = fabs(detj);

    gradients(cellOrdinal,0,0) = i11;
    gradients(cellOrdinal,1,0) =-i11;
  }

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeCellVolume
{
  private:
    Omega_h::Matrix<SpaceDim, SpaceDim> jacobian, jacobianInverse;

  public:
    template<typename ScalarType>
    DEVICE_TYPE inline void
    operator()(Plato::OrdinalType cellOrdinal,
               Plato::ScalarArray3DT<ScalarType> config,
               ScalarType& cellVolume) const;
};

  template<>
  template<typename ScalarType>
  DEVICE_TYPE inline void
  ComputeCellVolume<3>::operator()(Plato::OrdinalType cellOrdinal,
                                   Plato::ScalarArray3DT<ScalarType> config,
                                   ScalarType& cellVolume) const
  {
    ScalarType j11=config(cellOrdinal,0,0)-config(cellOrdinal,3,0);
    ScalarType j12=config(cellOrdinal,1,0)-config(cellOrdinal,3,0);
    ScalarType j13=config(cellOrdinal,2,0)-config(cellOrdinal,3,0);

    ScalarType j21=config(cellOrdinal,0,1)-config(cellOrdinal,3,1);
    ScalarType j22=config(cellOrdinal,1,1)-config(cellOrdinal,3,1);
    ScalarType j23=config(cellOrdinal,2,1)-config(cellOrdinal,3,1);

    ScalarType j31=config(cellOrdinal,0,2)-config(cellOrdinal,3,2);
    ScalarType j32=config(cellOrdinal,1,2)-config(cellOrdinal,3,2);
    ScalarType j33=config(cellOrdinal,2,2)-config(cellOrdinal,3,2);

    ScalarType detj = j11*j22*j33+j12*j23*j31+j13*j21*j32
                     -j11*j23*j32-j12*j21*j33-j13*j22*j31;

    cellVolume = fabs(detj);
  }

  template<>
  template<typename ScalarType>
  DEVICE_TYPE inline void
  ComputeCellVolume<2>::operator()(Plato::OrdinalType cellOrdinal,
                                   Plato::ScalarArray3DT<ScalarType> config,
                                   ScalarType& cellVolume) const
  {
    ScalarType j11=config(cellOrdinal,0,0)-config(cellOrdinal,2,0);
    ScalarType j12=config(cellOrdinal,1,0)-config(cellOrdinal,2,0);

    ScalarType j21=config(cellOrdinal,0,1)-config(cellOrdinal,2,1);
    ScalarType j22=config(cellOrdinal,1,1)-config(cellOrdinal,2,1);

    ScalarType detj = j11*j22-j12*j21;

    cellVolume = fabs(detj);
  }

  template<>
  template<typename ScalarType>
  DEVICE_TYPE inline void
  ComputeCellVolume<1>::operator()(Plato::OrdinalType cellOrdinal,
                                   Plato::ScalarArray3DT<ScalarType> config,
                                   ScalarType& cellVolume) const
  {
    ScalarType j11=config(cellOrdinal,0,0)-config(cellOrdinal,1,0);

    ScalarType detj = j11;

    cellVolume = fabs(detj);
  }


  /******************************************************************************/
  template<Plato::OrdinalType SpaceDim>
  class ComputeSurfaceArea
  {
    private:
      Omega_h::Matrix<SpaceDim, SpaceDim> jacobian, jacobianInverse;
    public:
      ComputeSurfaceArea(){}



      template<typename ScalarType>
      DEVICE_TYPE inline void
      operator()(Plato::OrdinalType aCellOrdinal,
                 int*               aCellLocalNodeOrdinals,
                 Plato::ScalarArray3DT<ScalarType> config,
                 ScalarType& sideArea) const;
  };


    template<>
    template<typename ScalarType>
    DEVICE_TYPE inline void
    ComputeSurfaceArea<3>::operator()(Plato::OrdinalType aCellOrdinal,
                                      int*                aCellLocalNodeOrdinals,
                                      Plato::ScalarArray3DT<ScalarType> config,
                                      ScalarType& sideArea) const
    {

        ScalarType ab0 = config(aCellOrdinal,aCellLocalNodeOrdinals[2],0) - config(aCellOrdinal,aCellLocalNodeOrdinals[0],0);
        ScalarType ab1 = config(aCellOrdinal,aCellLocalNodeOrdinals[2],1) - config(aCellOrdinal,aCellLocalNodeOrdinals[0],1);
        ScalarType ab2 = config(aCellOrdinal,aCellLocalNodeOrdinals[2],2) - config(aCellOrdinal,aCellLocalNodeOrdinals[0],2);

        ScalarType bc0 = config(aCellOrdinal,aCellLocalNodeOrdinals[1],0) - config(aCellOrdinal,aCellLocalNodeOrdinals[2],0);
        ScalarType bc1 = config(aCellOrdinal,aCellLocalNodeOrdinals[1],1) - config(aCellOrdinal,aCellLocalNodeOrdinals[2],1);
        ScalarType bc2 = config(aCellOrdinal,aCellLocalNodeOrdinals[1],2) - config(aCellOrdinal,aCellLocalNodeOrdinals[2],2);


        ScalarType Cross0 = ab1 * bc2 - ab2 * bc1;
        ScalarType Cross1 = ab2 * bc0 - ab0 * bc2;
        ScalarType Cross2 = ab0 * bc1 - ab1 * bc0;

        sideArea = sqrt(Cross0*Cross0 + Cross1*Cross1 + Cross2*Cross2)/2;
    }

    template<>
    template<typename ScalarType>
    DEVICE_TYPE inline void
    ComputeSurfaceArea<2>::operator()(Plato::OrdinalType aCellOrdinal,
                                      int*                aCellLocalNodeOrdinals,
                                      Plato::ScalarArray3DT<ScalarType> config,
                                      ScalarType& sideArea) const
    {
    }

    template<>
    template<typename ScalarType>
    DEVICE_TYPE inline void
    ComputeSurfaceArea<1>::operator()(Plato::OrdinalType aCellOrdinal,
                                      int*                aCellLocalNodeOrdinals,
                                      Plato::ScalarArray3DT<ScalarType> config,
                                      ScalarType& sideArea) const
    {
    }


/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeGradient
{
  private:
    const NodeCoordinate<SpaceDim> mNodeCoordinate;

  public:
    ComputeGradient(
      NodeCoordinate<SpaceDim> nodeCoordinate) :
      mNodeCoordinate(nodeCoordinate) {}

    DEVICE_TYPE
    inline
    void
    operator()(Plato::OrdinalType cellOrdinal,
               Omega_h::Vector<SpaceDim>* gradients,
               Scalar& cellVolume) const
    {
      // compute jacobian/Det/inverse for cell:
      //
      Omega_h::Matrix<SpaceDim, SpaceDim> jacobian, jacobianInverse;
      for (Plato::OrdinalType d1=0; d1<SpaceDim; d1++)
      {
        for (Plato::OrdinalType d2=0; d2<SpaceDim; d2++)
        {
          jacobian[d1][d2] = mNodeCoordinate(cellOrdinal,d2,d1) - mNodeCoordinate(cellOrdinal,SpaceDim,d1);
        }
      }
      Plato::Scalar jacobianDet = Omega_h::determinant(jacobian);
      jacobianInverse = Omega_h::invert(jacobian);
      cellVolume = fabs(jacobianDet);

      // ref gradients in 3D are:
      //    field 0 = ( 1 ,0, 0)
      //    field 1 = ( 0, 1, 0)
      //    field 2 = ( 0, 0, 1)
      //    field 3 = (-1,-1,-1)

      // Therefore, when we multiply by the transpose jacobian inverse (which is what we do to compute physical gradients),
      // we have the following values:
      //    field 0 = row 0 of jacobianInv
      //    field 1 = row 1 of jacobianInv
      //    field 2 = row 2 of jacobianInv
      //    field 3 = negative sum of the three rows

      for (Plato::OrdinalType d=0; d<SpaceDim; d++)
      {
        gradients[SpaceDim][d] = 0.0;
      }

      for (Plato::OrdinalType nodeOrdinal=0; nodeOrdinal<SpaceDim; nodeOrdinal++)  // "d1" for jacobian
      {
        for (Plato::OrdinalType d=0; d<SpaceDim; d++) // "d2" for jacobian
        {
          gradients[nodeOrdinal][d] = jacobianInverse[nodeOrdinal][d];
          gradients[SpaceDim][d]   -= jacobianInverse[nodeOrdinal][d];
        }
      }
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeVolume
{
  private:
    const NodeCoordinate<SpaceDim> mNodeCoordinate;

  public:
    ComputeVolume(
      NodeCoordinate<SpaceDim> nodeCoordinate) :
      mNodeCoordinate(nodeCoordinate) {}

    DEVICE_TYPE
    inline
	Scalar
    operator()(Plato::OrdinalType cellOrdinal) const
    {
      // compute jacobian/Det for cell:
      //
      Omega_h::Matrix<SpaceDim, SpaceDim> jacobian;
      for (Plato::OrdinalType d1=0; d1<SpaceDim; d1++)
      {
        for (Plato::OrdinalType d2=0; d2<SpaceDim; d2++)
        {
          jacobian[d1][d2] = mNodeCoordinate(cellOrdinal,d2,d1) - mNodeCoordinate(cellOrdinal,SpaceDim,d1);
        }
      }
      Plato::Scalar jacobianDet = Omega_h::determinant(jacobian);
      constexpr Plato::Scalar det2Vol = (SpaceDim == 2) ? 0.5 : 1./6.;
      return det2Vol*fabs(jacobianDet);
    }
};
/******************************************************************************/


/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeGradientMatrix : public Plato::SimplexMechanics<SpaceDim>
{
  private:

    using Plato::SimplexMechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexMechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::mNumDofsPerCell;
    static constexpr auto mNumSpaceDim = SpaceDim;

  public:

    DEVICE_TYPE
    void
    operator()( const Omega_h::Vector<mNumSpaceDim>* gradients,
                      Omega_h::Vector<mNumVoigtTerms>* gradientMatrix) const
    {
      for (Plato::OrdinalType iDof=0; iDof<mNumDofsPerCell; iDof++){
        for (Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
          gradientMatrix[iDof][iVoigt] = 0.0;
        }
      }

      for (Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++)
      {
        Plato::OrdinalType voigtTerm=0;
        for (Plato::OrdinalType iDof=0; iDof<mNumSpaceDim; iDof++){
          gradientMatrix[mNumSpaceDim*iNode+iDof][voigtTerm] = gradients[iNode][iDof];
          voigtTerm++;
        }

        for (Plato::OrdinalType jDof=mNumSpaceDim-1; jDof>=1; jDof--){
          for (Plato::OrdinalType iDof=jDof-1; iDof>=0; iDof--){
            gradientMatrix[mNumSpaceDim*iNode+iDof][voigtTerm] = gradients[iNode][jDof];
            gradientMatrix[mNumSpaceDim*iNode+jDof][voigtTerm] = gradients[iNode][iDof];
            voigtTerm++;
          }
        }
      }
    }
};
/******************************************************************************/


/******************************************************************************/
template<Plato::OrdinalType SpaceDim, typename OrdinalLookupType>
class Assemble
{
  private:
    static constexpr auto mNumVoigtTerms   = (SpaceDim == 3) ? 6 :
                                             ((SpaceDim == 2) ? 3 :
                                            (((SpaceDim == 1) ? 1 : 0)));
    static constexpr auto mNumNodesPerCell = SpaceDim+1;
    static constexpr auto mNumDofsPerCell  = SpaceDim*mNumNodesPerCell;

    const typename CrsMatrixType::ScalarVector mMatrixEntries;
    const Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    const OrdinalLookupType mEntryOrdinalLookup;
    const Plato::OrdinalType mEntriesLength;

  public:
    Assemble(const Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> aCellStiffness,
             Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
             OrdinalLookupType aEntryOrdinalLookup ) :
        mMatrixEntries(aMatrix->entries()),
        mCellStiffness(aCellStiffness),
        mEntryOrdinalLookup(aEntryOrdinalLookup),
        mEntriesLength(mMatrixEntries.size()) {}

    DEVICE_TYPE
    inline
    void
    operator()(Plato::OrdinalType cellOrdinal,
               const Omega_h::Vector<mNumVoigtTerms>* gradientMatrix,
               const Plato::Scalar& cellVolume) const
    {
      for (Plato::OrdinalType iDof=0; iDof<mNumDofsPerCell; iDof++)
      {
        for (Plato::OrdinalType jDof=0; jDof<mNumDofsPerCell; jDof++)
        {
            Plato::Scalar integral = (gradientMatrix[iDof] * (mCellStiffness * gradientMatrix[jDof])) * cellVolume;

            auto entryOrdinal = mEntryOrdinalLookup(cellOrdinal,iDof,jDof);
            if (entryOrdinal < mEntriesLength)
            {
                Kokkos::atomic_add(&mMatrixEntries(entryOrdinal), integral);
            }
        }
      }
    }
};
/******************************************************************************/


/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType DofsPerNode>
class BlockMatrixTransposeEntryOrdinal
{
  private:
    const typename CrsMatrixType::RowMapVector mRowMap;
    const typename CrsMatrixType::OrdinalVector mColumnIndices;
    const Omega_h::LOs mCells2nodes;

  public:
    BlockMatrixTransposeEntryOrdinal(Teuchos::RCP<Plato::CrsMatrixType> matrix, Omega_h::Mesh* mesh ) :
      mRowMap(matrix->rowMap()),
      mColumnIndices(matrix->columnIndices()),
      mCells2nodes(mesh->ask_elem_verts()) { }

    DEVICE_TYPE
    inline
    Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType icellDof, Plato::OrdinalType jcellDof) const
    {
        auto iNode = icellDof / DofsPerNode;
        auto iDof  = icellDof % DofsPerNode;
        auto jNode = jcellDof / DofsPerNode;
        auto jDof  = jcellDof % DofsPerNode;
        Plato::OrdinalType iLocalOrdinal = mCells2nodes[cellOrdinal * (SpaceDim+1) + iNode];
        Plato::OrdinalType jLocalOrdinal = mCells2nodes[cellOrdinal * (SpaceDim+1) + jNode];
        Plato::OrdinalType rowStart = mRowMap(jLocalOrdinal);
        Plato::OrdinalType rowEnd   = mRowMap(jLocalOrdinal+1);
        for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (mColumnIndices(entryOrdinal) == iLocalOrdinal)
          {
            return entryOrdinal*DofsPerNode*DofsPerNode+jDof*DofsPerNode+iDof;
          }
        }
        return Plato::OrdinalType(-1);
    }
};

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType BlockSize_I, Plato::OrdinalType BlockSize_J>
class LocalByGlobalEntryFunctor
{
  private:
    const typename CrsMatrixType::RowMapVector mRowMap;
    const typename CrsMatrixType::OrdinalVector mColumnIndices;
    const Omega_h::LOs mCells2nodes;

    using MatrixT = Teuchos::RCP<Plato::CrsMatrixType>;
    using MeshT   = Omega_h::Mesh*;

  public:
    LocalByGlobalEntryFunctor(
        MatrixT tMatrix,
        MeshT   tMesh
    ) :
      mRowMap        (tMatrix->rowMap()),
      mColumnIndices (tMatrix->columnIndices()),
      mCells2nodes   (tMesh->ask_elem_verts()) { }

    DEVICE_TYPE
    inline
    Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType icellDof, Plato::OrdinalType jcellDof) const
    {
        auto jNode = jcellDof / BlockSize_J;
        auto jDof  = jcellDof % BlockSize_J;
        Plato::OrdinalType jLocalOrdinal = mCells2nodes[cellOrdinal * (SpaceDim+1) + jNode];
        Plato::OrdinalType rowStart = mRowMap(cellOrdinal);
        Plato::OrdinalType rowEnd   = mRowMap(cellOrdinal+1);
        for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (mColumnIndices(entryOrdinal) == jLocalOrdinal)
          {
            return entryOrdinal*BlockSize_I*BlockSize_J + icellDof*BlockSize_J + jDof;
          }
        }
        return Plato::OrdinalType(-1);
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType DofsPerNode_I, Plato::OrdinalType DofsPerElem_J>
class GlobalByLocalEntryFunctor
{
  private:
    const typename CrsMatrixType::RowMapVector mRowMap;
    const typename CrsMatrixType::OrdinalVector mColumnIndices;
    const Omega_h::LOs mCells2nodes;

  public:
    GlobalByLocalEntryFunctor(Teuchos::RCP<Plato::CrsMatrixType> matrix, Omega_h::Mesh* mesh ) :
      mRowMap(matrix->rowMap()),
      mColumnIndices(matrix->columnIndices()),
      mCells2nodes(mesh->ask_elem_verts()) { }

    DEVICE_TYPE
    inline
    Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType icellDof, Plato::OrdinalType jcellDof) const
    {
        auto iNode = icellDof / DofsPerNode_I;
        auto iDof  = icellDof % DofsPerNode_I;
        Plato::OrdinalType iLocalOrdinal = mCells2nodes[cellOrdinal * (SpaceDim+1) + iNode];
        Plato::OrdinalType jLocalOrdinal = cellOrdinal * (SpaceDim+1) + jcellDof;
        Plato::OrdinalType rowStart = mRowMap(iLocalOrdinal);
        Plato::OrdinalType rowEnd   = mRowMap(iLocalOrdinal+1);
        for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (mColumnIndices(entryOrdinal) == jLocalOrdinal)
          {
            return entryOrdinal*DofsPerNode_I*DofsPerElem_J+iDof*DofsPerElem_J+jcellDof;
          }
        }
        return Plato::OrdinalType(-1);
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType DofsPerNode_I, Plato::OrdinalType DofsPerNode_J=DofsPerNode_I>
class BlockMatrixEntryOrdinal
{
  private:
    const typename CrsMatrixType::RowMapVector mRowMap;
    const typename CrsMatrixType::OrdinalVector mColumnIndices;
    const Omega_h::LOs mCells2nodes;

  public:
    BlockMatrixEntryOrdinal(Teuchos::RCP<Plato::CrsMatrixType> matrix, Omega_h::Mesh* mesh ) :
      mRowMap(matrix->rowMap()),
      mColumnIndices(matrix->columnIndices()),
      mCells2nodes(mesh->ask_elem_verts()) { }

    DEVICE_TYPE
    inline
    Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType icellDof, Plato::OrdinalType jcellDof) const
    {
        auto iNode = icellDof / DofsPerNode_I;
        auto iDof  = icellDof % DofsPerNode_I;
        auto jNode = jcellDof / DofsPerNode_J;
        auto jDof  = jcellDof % DofsPerNode_J;
        Plato::OrdinalType iLocalOrdinal = mCells2nodes[cellOrdinal * (SpaceDim+1) + iNode];
        Plato::OrdinalType jLocalOrdinal = mCells2nodes[cellOrdinal * (SpaceDim+1) + jNode];
        Plato::OrdinalType rowStart = mRowMap(iLocalOrdinal);
        Plato::OrdinalType rowEnd   = mRowMap(iLocalOrdinal+1);
        for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (mColumnIndices(entryOrdinal) == jLocalOrdinal)
          {
            return entryOrdinal*DofsPerNode_I*DofsPerNode_J+iDof*DofsPerNode_J+jDof;
          }
        }
        return Plato::OrdinalType(-1);
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofsPerNode_J=DofsPerNode>
class MatrixEntryOrdinal
{
  private:
    const typename CrsMatrixType::RowMapVector mRowMap;
    const typename CrsMatrixType::OrdinalVector mColumnIndices;
    const Omega_h::LOs mCells2nodes;

  public:
    MatrixEntryOrdinal(Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
                       Omega_h::Mesh* aMesh ) :
      mRowMap(aMatrix->rowMap()),
      mColumnIndices(aMatrix->columnIndices()),
      mCells2nodes(aMesh->ask_elem_verts()) { }

    DEVICE_TYPE
    inline
    Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType icellDof, Plato::OrdinalType jcellDof) const
    {
      auto iNode = icellDof / DofsPerNode;
      auto iDof  = icellDof % DofsPerNode;
      auto jNode = jcellDof / DofsPerNode;
      auto jDof  = jcellDof % DofsPerNode;
      Plato::OrdinalType iLocalOrdinal = mCells2nodes[cellOrdinal * (SpaceDim+1) + iNode];
      Plato::OrdinalType jLocalOrdinal = mCells2nodes[cellOrdinal * (SpaceDim+1) + jNode];
      Plato::OrdinalType rowStart = mRowMap(DofsPerNode*iLocalOrdinal+iDof  );
      Plato::OrdinalType rowEnd   = mRowMap(DofsPerNode*iLocalOrdinal+iDof+1);
      for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
      {
        if (mColumnIndices(entryOrdinal) == DofsPerNode*jLocalOrdinal+jDof)
        {
          return entryOrdinal;
        }
      }
      return Plato::OrdinalType(-1);
    }
};
/******************************************************************************/

/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  @param aMesh Omega_h mesh on which the matrix is based.

  Create a block matrix from connectivity in mesh with block size
  DofsPerNode_I X DofsPerElem_J.

  This function creates a matrix that stores a transpose of the gradient of
  local element states wrt nodal degrees of freedom.  Each column has the same
  number of non-zero block entries (NNodesPerCell)
*/
template <typename MatrixType, Plato::OrdinalType NodesPerElem, Plato::OrdinalType DofsPerNode_I, Plato::OrdinalType DofsPerElem_J>
Teuchos::RCP<MatrixType>
CreateGlobalByLocalBlockMatrix( Omega_h::Mesh* aMesh )
/******************************************************************************/
{
    auto node2elems = aMesh->ask_up(0, aMesh->dim());
    auto node2elems_map   = node2elems.a2ab;
    auto node2elems_elems = node2elems.ab2b;

    auto tNumElems = aMesh->nelems();
    auto tNumNodes = aMesh->nverts();
    auto tNumNonZeros = tNumElems*NodesPerElem;

    constexpr Plato::OrdinalType numBlockDofs = DofsPerNode_I*DofsPerElem_J;

    typename MatrixType::RowMapVector  rowMap        ("row map",        tNumNodes+1);
    typename MatrixType::ScalarVector  entries       ("matrix entries", tNumNonZeros*numBlockDofs);
    typename MatrixType::OrdinalVector columnIndices ("column indices", tNumNonZeros);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), LAMBDA_EXPRESSION(Plato::OrdinalType aNodeOrdinal)
    {
      auto tFrom = node2elems_map[aNodeOrdinal];
      auto tTo = node2elems_map[aNodeOrdinal+1];
      rowMap(aNodeOrdinal)   = tFrom;
      rowMap(aNodeOrdinal+1) = tTo;

      for( decltype(tFrom) tColumnEntry = tFrom; tColumnEntry < tTo; tColumnEntry++ )
      {
          columnIndices(tColumnEntry) = node2elems_elems[tColumnEntry];
      }
    });

    auto retMatrix = Teuchos::rcp(
     new MatrixType( rowMap, columnIndices, entries,
                     tNumNodes*DofsPerNode_I,
                     tNumElems*DofsPerElem_J,
                     DofsPerNode_I,
                     DofsPerElem_J )
    );
    return retMatrix;
}

/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  @param aMesh Omega_h mesh on which the matrix is based.

  Create a block matrix from connectivity in mesh with block size
  DofsPerElem_I X DofsPerNode_J.

  This function creates a matrix that stores a gradient of local element
  states wrt nodal degrees of freedom.  Each row has the same number of
  non-zero block entries (NNodesPerCell)
*/
template <typename MatrixType, Plato::OrdinalType NodesPerElem, Plato::OrdinalType DofsPerElem_I, Plato::OrdinalType DofsPerNode_J>
Teuchos::RCP<MatrixType>
CreateLocalByGlobalBlockMatrix( Omega_h::Mesh* aMesh )
/******************************************************************************/
{
    const auto& mCells2nodes = aMesh->ask_elem_verts();

    auto tNumElems = aMesh->nelems();
    auto tNumNonZeros = tNumElems*NodesPerElem;

    constexpr Plato::OrdinalType numBlockDofs = DofsPerElem_I*DofsPerNode_J;

    typename MatrixType::RowMapVector  rowMap        ("row map",        tNumElems+1);
    typename MatrixType::ScalarVector  entries       ("matrix entries", tNumNonZeros*numBlockDofs);
    typename MatrixType::OrdinalVector columnIndices ("column indices", tNumNonZeros);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumElems), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
      auto tFrom = aCellOrdinal*NodesPerElem;
      auto tTo   = (aCellOrdinal+1)*NodesPerElem;
      rowMap(aCellOrdinal)   = tFrom;
      rowMap(aCellOrdinal+1) = tTo;

      decltype(aCellOrdinal) tLocalIndex = 0;
      for( decltype(tFrom) tColumnEntry = tFrom; tColumnEntry < tTo; tColumnEntry++ )
      {
          columnIndices(tColumnEntry) = mCells2nodes[aCellOrdinal*NodesPerElem + tLocalIndex++];
      }
    });

    auto tNumNodes = aMesh->nverts();
    auto retMatrix = Teuchos::rcp(
     new MatrixType( rowMap, columnIndices, entries,
                     tNumElems*DofsPerElem_I, tNumNodes*DofsPerNode_J,
                     DofsPerElem_I, DofsPerNode_J )
    );
    return retMatrix;
}

/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  @param mesh Omega_h mesh on which the matrix is based.

  Create a block matrix from connectivity in mesh with block size
  DofsPerNode_I X DofsPerNode_J.
*/
template <typename MatrixType, Plato::OrdinalType DofsPerNode_I, Plato::OrdinalType DofsPerNode_J=DofsPerNode_I>
Teuchos::RCP<MatrixType>
CreateBlockMatrix( Omega_h::Mesh* mesh )
/******************************************************************************/
{
    const Plato::OrdinalType vertexDim = 0;
    Omega_h::Graph nodeNodeGraph = mesh->ask_star(vertexDim);

    auto rowMapOmega_h        = nodeNodeGraph.a2ab;
    auto columnIndicesOmega_h = nodeNodeGraph.ab2b;

    auto numRows = rowMapOmega_h.size() - 1;
    // Omega_h does not include the diagonals: add numRows, and then
    // add 1 to each rowMap entry after the first
    auto nnz = columnIndicesOmega_h.size() + numRows;

    // account for num dofs per node
    constexpr Plato::OrdinalType numBlockDofs = DofsPerNode_I*DofsPerNode_J;

    typename MatrixType::RowMapVector  rowMap("row map", numRows+1);
    typename MatrixType::ScalarVector  entries("matrix entries", nnz*numBlockDofs);
    typename MatrixType::OrdinalVector columnIndices("column indices", nnz);

    // The compressed row storage format in Omega_h doesn't include diagonals.  This
    // function creates a CRSMatrix with diagonal entries included.

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numRows), LAMBDA_EXPRESSION(Plato::OrdinalType rowNumber)
    {
      auto entryOffset_oh = rowMapOmega_h[rowNumber];
      auto R0 = rowMapOmega_h[rowNumber] + rowNumber;
      auto R1 = rowMapOmega_h[rowNumber+1] + rowNumber+1;
      auto numNodesThisRow = R1-R0;
      rowMap(rowNumber) = R0;
      rowMap(rowNumber+1) = R1;

      Plato::OrdinalType i_oh = 0; // will track i until we insert the diagonal entry
      for (Plato::OrdinalType i=0; i<numNodesThisRow; i_oh++, i++)
      {
        bool insertDiagonal = false;
        if ((i_oh == i) && (i_oh + entryOffset_oh >= rowMapOmega_h[rowNumber+1]))
        {
          // i_oh == i                    --> have not inserted diagonal
          // i_oh + entryOffset_oh > size --> at the end of the omega_h entries, should insert
          insertDiagonal = true;
        }
        else if (i_oh == i)
        {
          // i_oh + entryOffset_oh in bounds
          auto columnIndex = columnIndicesOmega_h[i_oh + entryOffset_oh];
          if (columnIndex > rowNumber)
          {
            insertDiagonal = true;
          }
        }
        if (insertDiagonal)
        {
          // store the diagonal entry
          columnIndices(R0+i) = rowNumber;
          i_oh--; // i_oh lags i by 1 after we hit the diagonal
        }
        else
        {
          columnIndices(R0+i) = columnIndicesOmega_h[i_oh + entryOffset_oh];
        }
      }
    });

    auto retMatrix = Teuchos::rcp(
     new MatrixType( rowMap, columnIndices, entries,
                     numRows*DofsPerNode_I, numRows*DofsPerNode_J,
                     DofsPerNode_I, DofsPerNode_J )
    );
    return retMatrix;
}

/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  @param mesh Omega_h mesh on which the matrix is based.

  Create a matrix from connectivity in mesh with DofsPerNode.
*/
template <typename MatrixType, Plato::OrdinalType DofsPerNode>
Teuchos::RCP<MatrixType>
CreateMatrix( Omega_h::Mesh* mesh )
/******************************************************************************/
{
    const Plato::OrdinalType vertexDim = 0;
    Omega_h::Graph nodeNodeGraph = mesh->ask_star(vertexDim);

    auto rowMapOmega_h        = nodeNodeGraph.a2ab;
    auto columnIndicesOmega_h = nodeNodeGraph.ab2b;

    auto numRows = rowMapOmega_h.size() - 1;
    // Omega_h does not include the diagonals: add numRows, and then
    // add 1 to each rowMap entry after the first
    auto nnz = columnIndicesOmega_h.size() + numRows;

    // account for num dofs per node
    constexpr Plato::OrdinalType numDofsSquared = DofsPerNode*DofsPerNode;

    typename MatrixType::RowMapVector  rowMap("row map", numRows*DofsPerNode+1);
    typename MatrixType::ScalarVector  entries("matrix entries", nnz*numDofsSquared);
    typename MatrixType::OrdinalVector columnIndices("column indices", nnz*numDofsSquared);

    // The compressed row storage format in Omega_h doesn't include diagonals.  This
    // function creates a CRSMatrix with diagonal entries included and expands the
    // graph to DofsPerNode.

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numRows), LAMBDA_EXPRESSION(Plato::OrdinalType rowNumber)
    {
      auto entryOffset_oh = rowMapOmega_h[rowNumber];
      auto R0 = rowMapOmega_h[rowNumber] + rowNumber;
      auto R1 = rowMapOmega_h[rowNumber+1] + rowNumber+1;
      auto numNodesThisRow = R1-R0;
      auto numDofsThisRow = numNodesThisRow*DofsPerNode;
      auto dofRowOffset = DofsPerNode*rowNumber;
      auto dofColOffset = numDofsSquared*R0;
      for (Plato::OrdinalType iDof=0; iDof<=DofsPerNode; iDof++){
        rowMap(dofRowOffset+iDof) = dofColOffset+iDof*numDofsThisRow;
      }

      Plato::OrdinalType i_oh = 0; // will track i until we insert the diagonal entry
      for (Plato::OrdinalType i=0; i<numNodesThisRow; i_oh++, i++)
      {
        bool insertDiagonal = false;
        if ((i_oh == i) && (i_oh + entryOffset_oh >= rowMapOmega_h[rowNumber+1]))
        {
          // i_oh == i                    --> have not inserted diagonal
          // i_oh + entryOffset_oh > size --> at the end of the omega_h entries, should insert
          insertDiagonal = true;
        }
        else if (i_oh == i)
        {
          // i_oh + entryOffset_oh in bounds
          auto columnIndex = columnIndicesOmega_h[i_oh + entryOffset_oh];
          if (columnIndex > rowNumber)
          {
            insertDiagonal = true;
          }
        }
        if (insertDiagonal)
        {
          // store the diagonal entry
          for (Plato::OrdinalType iDof=0; iDof<DofsPerNode; iDof++){
            columnIndices(numDofsSquared*R0+DofsPerNode*i+iDof) = DofsPerNode*rowNumber+iDof;
          }
          i_oh--; // i_oh lags i by 1 after we hit the diagonal
        }
        else
        {
          for (Plato::OrdinalType iDof=0; iDof<DofsPerNode; iDof++){
            columnIndices(dofColOffset+DofsPerNode*i+iDof) = DofsPerNode*columnIndicesOmega_h[i_oh + entryOffset_oh]+iDof;
          }
        }
      }
      for (Plato::OrdinalType iDof=0; iDof<numDofsThisRow; iDof++)
      {
        for (Plato::OrdinalType jDof=1; jDof<DofsPerNode; jDof++){
          columnIndices(dofColOffset+jDof*numDofsThisRow+iDof) = columnIndices(dofColOffset+iDof);
        }
      }
    });

    auto retMatrix = Teuchos::rcp(new MatrixType( rowMap, columnIndices, entries ));
    return retMatrix;
}


} // end namespace Plato

#endif
