#ifndef PLATO_LINEAR_STRESS_HPP
#define PLATO_LINEAR_STRESS_HPP

#include "AbstractLinearStress.hpp"

namespace Plato
{
/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}
 */
/******************************************************************************/
template< typename EvaluationType, typename SimplexPhysics >
class LinearStress :
    public Plato::AbstractLinearStress<EvaluationType, SimplexPhysics>
{
protected:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using StateT  = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    using StrainT = typename Plato::fad_type_t<SimplexPhysics, StateT, ConfigT>; /*!< strain variables automatic differentiation type */

    using Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of stress/strain terms */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
    **********************************************************************************/
    LinearStress(const Omega_h::Matrix<mNumVoigtTerms,
                 mNumVoigtTerms> aCellStiffness) :
      AbstractLinearStress< EvaluationType, SimplexPhysics >(aCellStiffness)
    {
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model interface
    **********************************************************************************/
    LinearStress(const Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> aMaterialModel) :
      AbstractLinearStress< EvaluationType, SimplexPhysics >(aMaterialModel)
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
     * \param [in]  aCellOrdinal element ordinal
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
    **********************************************************************************/
    DEVICE_TYPE inline void operator()(Plato::OrdinalType aCellOrdinal,
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
};
// class LinearStress

}// namespace Plato
#endif


#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC2(Plato::LinearStress, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC2(Plato::LinearStress, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC2(Plato::LinearStress, Plato::SimplexMechanics, 3)
#endif
