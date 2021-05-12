#ifndef PLATO_HYPERBOLIC_LINEAR_STRESS_HPP
#define PLATO_HYPERBOLIC_LINEAR_STRESS_HPP

#include "LinearStress.hpp"

#include "hyperbolic/AbstractHyperbolicLinearStress.hpp"

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}

 Note HyperbolicLinearStress has TWO parent classes.  By having
 LinearStress as a parent the HyperbolicLinearStress can call the
 original LinearStress operator (sans VelGrad) or the operator with
 VelGrad as defined in AbstractHyperbolicLinearStress. That is the
 HyperbolicLinearStress contains both operator interfaces.

 */
/******************************************************************************/
template< typename EvaluationType, typename SimplexPhysics >
class HyperbolicLinearStress :
    public Plato::Hyperbolic::AbstractHyperbolicLinearStress<EvaluationType, SimplexPhysics>,
    public Plato::LinearStress<EvaluationType, SimplexPhysics>
{
protected:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using StateT    = typename EvaluationType::StateScalarType;    /*!< state variables automatic differentiation type */
    using StateDotT = typename EvaluationType::StateDotScalarType; /*!< state dot variables automatic differentiation type */
    using ConfigT   = typename EvaluationType::ConfigScalarType;   /*!< configuration variables automatic differentiation type */
    using ResultT   = typename EvaluationType::ResultScalarType;   /*!< result variables automatic differentiation type */

    using StrainT  = typename Plato::fad_type_t<SimplexPhysics, StateT,    ConfigT>; /*!<   strain variables automatic differentiation type */
    using VelGradT = typename Plato::fad_type_t<SimplexPhysics, StateDotT, ConfigT>; /*!< vel grad variables automatic differentiation type */

    using Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of stress/strain terms */

public:

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
    **********************************************************************************/
    HyperbolicLinearStress(const Omega_h::Matrix<mNumVoigtTerms,
                 mNumVoigtTerms> aCellStiffness) :
      AbstractHyperbolicLinearStress< EvaluationType, SimplexPhysics >(aCellStiffness),
      LinearStress< EvaluationType, SimplexPhysics >(aCellStiffness)
    {
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model interface
    **********************************************************************************/
    HyperbolicLinearStress(const Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> aMaterialModel) :
      AbstractHyperbolicLinearStress< EvaluationType, SimplexPhysics >(aMaterialModel),
      LinearStress< EvaluationType, SimplexPhysics >(aMaterialModel)
    {
    }

    // Make sure the original operator from LinearStress (sans
    // aVelGrad) is still visible. That is the operator() is
    // overloaded rather being overridden by the new method defined
    // below that includes the velosity gradient (aVelGrad).
    using LinearStress<EvaluationType, SimplexPhysics>::operator();

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
     * \param [in]  aVelGrad Velocity gradient tensor
    **********************************************************************************/
    void
    operator()(Plato::ScalarMultiVectorT<ResultT > const& aCauchyStress,
               Plato::ScalarMultiVectorT<StrainT > const& aSmallStrain,
               Plato::ScalarMultiVectorT<VelGradT> const& aVelGrad) const override
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
      const auto tRayleighB       = this->mRayleighB;
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
                ((aSmallStrain(aCellOrdinal, tVoigtIndex_J) - tReferenceStrain(tVoigtIndex_J)) +
                 (aVelGrad(aCellOrdinal, tVoigtIndex_J) * tRayleighB)) *
                tCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
          }
      } );
    }

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [in]  aCellOrdinal element ordinal
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
     * \param [in]  aVelGrad Velocity gradient tensor
    **********************************************************************************/
    DEVICE_TYPE inline void operator()(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarMultiVectorT<ResultT > const& aCauchyStress,
                                       Plato::ScalarMultiVectorT<StrainT > const& aSmallStrain,
                                       Plato::ScalarMultiVectorT<VelGradT> const& aVelGrad) const
    {
        // Method used to compute the stress and called from within a
        // Kokkos parallel_for.
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            aCauchyStress(aCellOrdinal, tVoigtIndex_I) = 0.0;

            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
                aCauchyStress(aCellOrdinal, tVoigtIndex_I) +=
                  ((aSmallStrain(aCellOrdinal, tVoigtIndex_J) - this->mReferenceStrain(tVoigtIndex_J)) +
                   (aVelGrad(aCellOrdinal, tVoigtIndex_J) * this->mRayleighB)) *
                  this->mCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
    }
};
// class HyperbolicLinearStress

}// namespace Hyperbolic

}// namespace Plato

#endif

#ifdef PLATOANALYZE_1D
  PLATO_HYPERBOLIC_EXPL_DEC2(Plato::Hyperbolic::HyperbolicLinearStress, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
  PLATO_HYPERBOLIC_EXPL_DEC2(Plato::Hyperbolic::HyperbolicLinearStress, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
  PLATO_HYPERBOLIC_EXPL_DEC2(Plato::Hyperbolic::HyperbolicLinearStress, Plato::SimplexMechanics, 3)
#endif
