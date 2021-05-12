#ifndef PLATO_ELLIPTIC_UPDATED_LAGRANGIAN_LINEAR_STRESS_HPP
#define PLATO_ELLIPTIC_UPDATED_LAGRANGIAN_LINEAR_STRESS_HPP

#include "AbstractEllipticUpLagLinearStress.hpp"

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}
 */
/******************************************************************************/
template< typename EvaluationType, typename SimplexPhysics >
class EllipticUpLagLinearStress :
    public Plato::Elliptic::UpdatedLagrangian::AbstractEllipticUpLagLinearStress<EvaluationType, SimplexPhysics>
{
protected:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using GlobalStateT = typename EvaluationType::GlobalStateScalarType; /*!< state variables automatic differentiation type */
    using LocalStateT  = typename EvaluationType::LocalStateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigT      = typename EvaluationType::ConfigScalarType;      /*!< configuration variables automatic differentiation type */
    using ResultT      = typename EvaluationType::ResultScalarType;      /*!< result variables automatic differentiation type */

    using StrainT = typename Plato::fad_type_t<SimplexPhysics, GlobalStateT, ConfigT>; /*!< strain variables automatic differentiation type */

    using Plato::Elliptic::UpdatedLagrangian::SimplexMechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of stress/strain terms */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
    **********************************************************************************/
    EllipticUpLagLinearStress(const Omega_h::Matrix<mNumVoigtTerms,
                 mNumVoigtTerms> aCellStiffness) :
      AbstractEllipticUpLagLinearStress< EvaluationType, SimplexPhysics >(aCellStiffness)
    {
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model interface
    **********************************************************************************/
    EllipticUpLagLinearStress(const Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> aMaterialModel) :
      AbstractEllipticUpLagLinearStress< EvaluationType, SimplexPhysics >(aMaterialModel)
    {
    }

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
    **********************************************************************************/
    void
    operator()(Plato::ScalarMultiVectorT<ResultT> const& aCauchyStress,
               Plato::ScalarMultiVectorT<StrainT> const& aSmallStrain) const override
  {
       // Method used to compute the stress with the factory and has
       // its own Kokkos parallel_for.

       // A lambda inside a member function captures the "this"
       // pointer not the actual members as such a local copy of the
       // data is need here for the lambda to capture everything.

       // If compiling with C++17 (Clang as the compiler or CUDA 11
       // with Kokkos 3.2). And using KOKKOS_CLASS_LAMBDA instead of
       // KOKKOS_EXPRESSION. Then the memeber data can be used
       // directly.
      const auto tCellStiffness   = this->mCellStiffness;
      const auto tReferenceStrain = this->mReferenceStrain;

      const Plato::OrdinalType tNumCells = aCauchyStress.extent(0);

      // Because the parallel_for loop is local, two dimensions of
      // parallelism can be exploited.
      Kokkos::parallel_for("Compute linear stress",
                           Kokkos::MDRangePolicy< Kokkos::Rank<2> >( {0, 0}, {tNumCells, mNumVoigtTerms} ),
                           LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal,
                                             const Plato::OrdinalType & tVoigtIndex_I)
      {
          aCauchyStress(aCellOrdinal, tVoigtIndex_I) = 0.0;

          for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
          {
              aCauchyStress(aCellOrdinal, tVoigtIndex_I) +=
                (aSmallStrain(aCellOrdinal, tVoigtIndex_J) -
                  tReferenceStrain(tVoigtIndex_J)) *
                tCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
          }
      } );
    }

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aStrainInc Incremental strain tensor
     * \param [in]  aPrevStrain Reference strain tensor
    **********************************************************************************/
    void
    operator()(Plato::ScalarMultiVectorT<ResultT>     const& aCauchyStress,
               Plato::ScalarMultiVectorT<StrainT>     const& aStrainInc,
               Plato::ScalarMultiVectorT<LocalStateT> const& aPrevStrain) const override
  {
       // Method used to compute the stress with the factory and has
       // its own Kokkos parallel_for.

       // A lambda inside a member function captures the "this"
       // pointer not the actual members as such a local copy of the
       // data is need here for the lambda to capture everything.

       // If compiling with C++17 (Clang as the compiler or CUDA 11
       // with Kokkos 3.2). And using KOKKOS_CLASS_LAMBDA instead of
       // KOKKOS_EXPRESSION. Then the memeber data can be used
       // directly.
      const auto tCellStiffness   = this->mCellStiffness;
      const auto tReferenceStrain = this->mReferenceStrain;

      const Plato::OrdinalType tNumCells = aCauchyStress.extent(0);

      // Because the parallel_for loop is local, two dimensions of
      // parallelism can be exploited.
      Kokkos::parallel_for("Compute linear stress",
                           Kokkos::MDRangePolicy< Kokkos::Rank<2> >( {0, 0}, {tNumCells, mNumVoigtTerms} ),
                           LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal,
                                             const Plato::OrdinalType & tVoigtIndex_I)
      {
          aCauchyStress(aCellOrdinal, tVoigtIndex_I) = 0.0;

          for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
          {
            aCauchyStress(aCellOrdinal, tVoigtIndex_I) +=
              (aStrainInc(aCellOrdinal, tVoigtIndex_J) -
               tReferenceStrain(tVoigtIndex_J) +
               aPrevStrain(aCellOrdinal, tVoigtIndex_J)) *
              tCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
          }
      } );
    }

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [in]  aCellOrdinal element ordinal
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
    **********************************************************************************/
    DEVICE_TYPE inline void
    operator()(Plato::OrdinalType aCellOrdinal,
               Plato::ScalarMultiVectorT<ResultT> const& aCauchyStress,
               Plato::ScalarMultiVectorT<StrainT> const& aSmallStrain) const
    {
        // Method used to compute the stress and called from within a
        // Kokkos parallel_for.
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            aCauchyStress(aCellOrdinal, tVoigtIndex_I) = 0.0;

            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
                aCauchyStress(aCellOrdinal, tVoigtIndex_I) +=
                  (aSmallStrain(aCellOrdinal, tVoigtIndex_J) -
                   this->mReferenceStrain(tVoigtIndex_J)) *
                  this->mCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
    }

    /******************************************************************************//**
     * \brief Compute Cauchy stress tensor
     * \param [in]  aCellOrdinal element ordinal
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aStrainInc Incremental strain tensor
     * \param [in]  aPrevStrain Reference strain tensor
    **********************************************************************************/
    DEVICE_TYPE inline void
    operator()(Plato::OrdinalType aCellOrdinal,
               Plato::ScalarMultiVectorT<ResultT>     const& aCauchyStress,
               Plato::ScalarMultiVectorT<StrainT>     const& aStrainInc,
               Plato::ScalarMultiVectorT<LocalStateT> const& aPrevStrain) const
    {
        // compute stress
        //
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            aCauchyStress(aCellOrdinal, tVoigtIndex_I) = 0.0;

            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
                aCauchyStress(aCellOrdinal, tVoigtIndex_I) +=
                    (aStrainInc(aCellOrdinal, tVoigtIndex_J) -
                     this->mReferenceStrain(tVoigtIndex_J) +
                     aPrevStrain(aCellOrdinal, tVoigtIndex_J)) *
                  this->mCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
    }
};
// class EllipticUpLagLinearStress

}// namespace UpdatedLagrangian

}// namespace Elliptic

}// namespace Plato
#endif


#ifdef PLATOANALYZE_1D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC2(Plato::Elliptic::UpdatedLagrangian::EllipticUpLagLinearStress, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC2(Plato::Elliptic::UpdatedLagrangian::EllipticUpLagLinearStress, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC2(Plato::Elliptic::UpdatedLagrangian::EllipticUpLagLinearStress, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 3)
#endif
