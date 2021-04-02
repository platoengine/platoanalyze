#ifndef PLATO_HYPERBOLIC_LINEAR_STRESS_EXPRESSION_HPP
#define PLATO_HYPERBOLIC_LINEAR_STRESS_EXPRESSION_HPP

#include "LinearStressExpression.hpp"

#include "ExpressionEvaluator.hpp"
#include "ParseTools.hpp"

#include "hyperbolic/HyperbolicAbstractLinearStress.hpp"

/******************************************************************************/
/*
 To use the expression evaluator one must first add an expression to
 the analyzeInput.xml file as part of the 'Custom Elasticity Model':

      <ParameterList name='Custom Elasticity Model'>
        <Parameter  name='Equation' type='string' value='CellStiffness * (SmallStrain - ReferenceStrain)'/>

        <Parameter  name='SmallStrain'   type='string' value='strain'/>

 Here the equation variable names will be mapped to the parameter
 labels which are which are currently fixed to be ReferenceStrain, and
 CellStiffness as they are class member variables and have not labels.

 Whereas the equation variable name, SmallStrain is indirectly mapped
 to the parameter label because the parameter label is not very
 descriptive.

 Note: SmallStrain is the state varaible, aSmallStrain must be
 passed in to the operator() regardless it if used or not. Whereas the
 ReferenceStrain and CellStiffness parameters, though set in the
 constructor.

 Equation variables can also be fixed values:
        <Parameter  name='stiffness' type='double' value='0.01'/>

 The equation can also be from a Bingo file:

      <ParameterList name='Custom Elasticity Model'>
        <Parameter name="BingoFile" type="string" value="bingo.txt"/>

*/
/******************************************************************************/

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}

 Note HyperbolicLinearStressExpression has TWO parent classes.  By
 having LinearStressExpression as a parent the
 HyperbolicLinearStressExpression can call the original
 LinearStressExpression operator (sans VelGrad) or the operator with
 VelGrad as defined in HyperbolicAbstractLinearStress. That is the
 HyperbolicLinearStressExpression contains both operator interfaces.

 */
/******************************************************************************/
template< typename EvaluationType, typename SimplexPhysics >
class HyperbolicLinearStressExpression :
    public Plato::Hyperbolic::HyperbolicAbstractLinearStress<EvaluationType, SimplexPhysics>,
    public Plato::LinearStressExpression<EvaluationType, SimplexPhysics>
{
protected:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using StateT    = typename EvaluationType::StateScalarType;     /*!< state variables automatic differentiation type */
    using StateDotT = typename EvaluationType::StateDotScalarType;
    using ConfigT   = typename EvaluationType::ConfigScalarType;    /*!< configuration variables automatic differentiation type */
    using ResultT   = typename EvaluationType::ResultScalarType;    /*!< result variables automatic differentiation type */

    using StrainT  = typename Plato::fad_type_t<SimplexPhysics, StateT,    ConfigT>; /*!<   strain variables automatic differentiation type */
    using VelGradT = typename Plato::fad_type_t<SimplexPhysics, StateDotT, ConfigT>; /*!< vel grad variables automatic differentiation type */

    using Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms;               /*!< number of stress/strain terms */

    Teuchos::ParameterList mInputParams;

public:
// ************************************************************************* //
    // Map structure - used with Kokkos so char strings so to be compatable.
    template< typename KEY_TYPE, typename VALUE_TYPE > struct _Map {
      KEY_TYPE key;
      VALUE_TYPE value;
    };

    template< typename KEY_TYPE, typename VALUE_TYPE >
    using Map = _Map< KEY_TYPE, VALUE_TYPE>;

    using VariableMap = Map< Plato::OrdinalType, char[MAX_ARRAY_LENGTH] >;

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    HyperbolicLinearStressExpression(const Omega_h::Matrix<mNumVoigtTerms,
                                     mNumVoigtTerms> aCellStiffness,
                                     const Teuchos::ParameterList& aInputParams) :
      HyperbolicAbstractLinearStress< EvaluationType, SimplexPhysics >(aCellStiffness),
      LinearStressExpression< EvaluationType, SimplexPhysics >(aCellStiffness, aInputParams),
      mInputParams(aInputParams)
    {
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model interface
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    HyperbolicLinearStressExpression(const Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> aMaterialModel,
                                     const Teuchos::ParameterList& aInputParams) :
      HyperbolicAbstractLinearStress< EvaluationType, SimplexPhysics >(aMaterialModel),
      LinearStressExpression< EvaluationType, SimplexPhysics >(aMaterialModel, aInputParams),
      mInputParams(aInputParams)
    {
    }

    // Make sure the original operator from LinearStressExpression
    // (sans aVelGrad) is still visible. That is the operator() is
    // overloaded rather being overridden by the new method defined
    // below that includes the velosity gradient (aVelGrad).
    using LinearStressExpression<EvaluationType, SimplexPhysics>::operator();

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
      // Method used with the factory and has it own Kokkos parallel_for
      const Plato::OrdinalType tNumCells = aCauchyStress.extent(0);

      // A lambda inside a member function captures the "this"
      // pointer not the actual members as such a local copy of the
      // data is need here for the lambda to capture everything.

      // If compiling with C++17 (Clang as the compiler or CUDA 11
      // with Kokkos 3.2). And using KOKKOS_CLASS_LAMBDA instead of
      // KOKKOS_EXPRESSION. Then the memeber data can be used
      // directly.
      // const auto tCellStiffness   = this->mCellStiffness;
      // const auto tReferenceStrain = this->mReferenceStrain;

      // Because a view of views is used in the expression which are
      // reference counted and deleting the parent view DOES NOT
      // de-reference so do not use the Omega_h structures
      // directly. Instead use a Kokkos::view and make a local copy
      // which is needed anyways for the reasons above, that view can
      // be re-referenced directly.
      Plato::ScalarVectorT<Plato::Scalar> tCellStiffness
        ("Temporary Cell Stiffness", mNumVoigtTerms, mNumVoigtTerms);
      Plato::ScalarVectorT<Plato::Scalar> tReferenceStrain
        ("Temporary Reference Strain", mNumVoigtTerms);

      Kokkos::parallel_for("Creating a local copy",
                           Kokkos::RangePolicy<>(0, mNumVoigtTerms),
                           LAMBDA_EXPRESSION(const Plato::OrdinalType & iIndex)
      {
        tReferenceStrain(iIndex) = this->mReferenceStrain(iIndex);

        for(Plato::OrdinalType jIndex = 0; jIndex < mNumVoigtTerms; jIndex++)
        {
          tCellStiffness(iIndex, jIndex) = this->mCellStiffness(iIndex, jIndex);
        }
      } );

      // The expression evaluator has a limited number of types so
      // convert the VelGrad and the Strain to the result type. This
      // conversion will often be redundant for one or both the
      // variables. But it is the only way currently to assure both
      // are the same type.
      Plato::ScalarMultiVectorT<ResultT>
        tVelGrad    (aVelGrad.label(),     tNumCells, mNumVoigtTerms),
        tSmallStrain(aSmallStrain.label(), tNumCells, mNumVoigtTerms);

      Kokkos::parallel_for("Convert vel grad and strain to common state type",
                           Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,mNumVoigtTerms}),
                           LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal,
                                             const Plato::OrdinalType & tVoigtIndex)
      {
          // Convert the vel grad and strain to the common state type.
          tVelGrad    (aCellOrdinal, tVoigtIndex) = aVelGrad    (aCellOrdinal, tVoigtIndex);
          tSmallStrain(aCellOrdinal, tVoigtIndex) = aSmallStrain(aCellOrdinal, tVoigtIndex);
      } );

      // Indices for the equation variable mapping.
      const Plato::OrdinalType cRayleighB       = 0;
      const Plato::OrdinalType cCellStiffness   = 1;
      const Plato::OrdinalType cReferenceStrain = 2;
      const Plato::OrdinalType cVelGrad         = 3;
      const Plato::OrdinalType cSmallStrain     = 4; // local state last
      const Plato::OrdinalType tNumParamLabels  = 5;

      // Strings for mapping parameter labels to the equation
      // variables. The CellStiffness and ReferenceStrain are class
      // member variables and have fixed names whereas the
      // Velgrad and SmallStrain are a required parameters.
      std::vector< std::string > tParamLabels( tNumParamLabels );

      tParamLabels[cRayleighB]       = "RayleighB";
      tParamLabels[cCellStiffness]   = "CellStiffness";
      tParamLabels[cReferenceStrain] = "ReferenceStrain";
      tParamLabels[cVelGrad]         = aVelGrad.label();
      tParamLabels[cSmallStrain]     = aSmallStrain.label();

      // If the user wants to use the input parameters these hold the
      // names of the equation variables that are mapped to the input
      // parameter labels.
      Kokkos::View< VariableMap *, Kokkos::CudaUVMSpace >
        tVarMaps ("Linear Stress Exp. Variable Maps", tNumParamLabels);

      // No mappings initially.
      for( Plato::OrdinalType i=0; i<tNumParamLabels; ++i )
        tVarMaps(i).key = 0;

      /*!< expression evaluator */
      ExpressionEvaluator< Plato::ScalarMultiVectorT<ResultT>,
                           Plato::ScalarMultiVectorT<ResultT>,
                           Plato::ScalarVectorT<Plato::Scalar>,
                           // Omega_h::Vector<mNumVoigtTerms>,
                           Plato::Scalar > tExpEval;

      // Look for a Custom Elasticity Model
      if( mInputParams.isSublist("Custom Elasticity Model") )
      {
        auto tCPMParams = mInputParams.sublist("Custom Elasticity Model");

        // Get the expression from the parameters.
        std::string tEquationStr = ParseTools::getEquationParam(tCPMParams);

        // Parse the expression.
        tExpEval.parse_expression(tEquationStr.c_str());

        // For all of the variables found in the expression optionally
        // get their values from the parameter list.
        const std::vector< std::string > tVarNames =
          tExpEval.get_variables();

        for( auto const & tVarName : tVarNames )
        {
          // Here the expression variable is found as a Plato::Scalar
          // so the value comes from the xml and is set directly.
          if( tCPMParams.isType<Plato::Scalar>(tVarName) )
          {
            // The value *MUST BE* converted to the Plato::Scalar as it is
            // the type used for all fixed variables.
            Plato::Scalar tVal = tCPMParams.get<Plato::Scalar>(tVarName);

            tExpEval.set_variable( tVarName.c_str(), tVal );
          }
          // Here the expression variable is found as a string so the
          // values should come from the parameters passed in.
          else if( tCPMParams.isType<std::string>(tVarName) )
          {
            std::string tVal = tCPMParams.get<std::string>(tVarName);

            // These are the labels of the parameters passed into the
            // evaluation operator below. If the equation variable
            // "value" matches then the parameter value will be used.
            bool tFound = false;

            for( Plato::OrdinalType i=0; i<tNumParamLabels; ++i )
            {
              if( tVal == tParamLabels[i] )
              {
                tVarMaps(i).key = 1;
                strcpy( tVarMaps(i).value, tVarName.c_str() );

                tFound = true;
                break;
              }
            }

            if( !tFound )
            {
              std::stringstream errorMsg;
              errorMsg << "Invalid parameter name '" << tVal << "' "
                       << "found for varaible name '" << tVarName << "'. "
                       << "It must be :";

              for( Plato::OrdinalType i=0; i<tNumParamLabels; i++ )
              {
                errorMsg << " '" << tParamLabels[i] << "'";
              }

              errorMsg << ".";

             THROWERR(  errorMsg.str() );
            }
          }
          // Here the expression variable should come from the
          // parameters passed in.
          else
          {
            // These are the names of the parameters passed into the
            // evaluation operator below. If the equation variable
            // name matches then the parameter value will be used.
            bool tFound = false;

            for( Plato::OrdinalType i=0; i<tNumParamLabels; ++i )
            {
              if( tVarName == tParamLabels[i] )
              {
                tVarMaps(i).key = 1;
                strcpy( tVarMaps(i).value, tVarName.c_str() );

                tFound = true;
                break;
              }
            }

            if( !tFound )
            {
              std::stringstream errorMsg;
              errorMsg << "Invalid varaible name '" << tVarName << "'. "
                       << "It must be :";

              for( Plato::OrdinalType i=0; i<tNumParamLabels; i++ )
              {
                errorMsg << " '" << tParamLabels[i] << "'";
              }

              errorMsg << ".";

             THROWERR(  errorMsg.str() );
            }
          }
        }
      }

      // If for some reason the expression evalutor is called but
      // without the XML block.
      else
      {
        THROWERR("Warning: Failed to find a 'Custom Elasticity Model' block.");
      }

      // After the parsing, set up the storage the sizes must match
      // the input and output data sizes.
      tExpEval.setup_storage( tNumCells, mNumVoigtTerms );

      // Input values which is a two-dimensional array. The first
      // index is over the cell index. The second index is over tVoigtIndex_J.
      if( tVarMaps(cSmallStrain).key )
      {
          tExpEval.set_variable( tVarMaps(cSmallStrain).value, tSmallStrain );
      }

      // Additional input values also a two-dimensional arry.
      if( tVarMaps(cVelGrad).key )
      {
          tExpEval.set_variable( tVarMaps(cVelGrad).value, tVelGrad );
      }

      // The reference strain does not change.
      if( tVarMaps(cReferenceStrain).key )
      {
        tExpEval.set_variable( tVarMaps(cReferenceStrain).value, tReferenceStrain );
      }

      // The RayleighB does not change.
      if( tVarMaps(cRayleighB).key )
        tExpEval.set_variable( tVarMaps(cRayleighB).value, this->mRayleighB );

      // Temporary memory for the stress that is returned from the
      // expression evaluation. The second index is over tVoigtIndex_J.
      Plato::ScalarMultiVectorT<ResultT> tStress("Temporary Linear Stress",
                                                 tNumCells, mNumVoigtTerms);

      // Note: unlike the original parallel_for one dimension of
      // parallelism is lost because at present the expression
      // evaluation is over a single parallel index.
      Kokkos::parallel_for("Compute linear stress",
                           Kokkos::RangePolicy<>(0, tNumCells),
                           LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        // Compute the stress.  This loop cannot be parallelized
        // because the cell stiffness is set locally and is used by
        // all threads. In other words the tCellStiffness[tVoigtIndex_I]
        // is in shared memory and used by all threads
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            // Values that change based on the tVoigtIndex_I index.
            if( tVarMaps(cCellStiffness).key )
              tExpEval.set_variable( tVarMaps(cCellStiffness).value,
                                     tCellStiffness[tVoigtIndex_I],
                                     aCellOrdinal );

            // Evaluate the expression for this cell. Note: the second
            // index of tStress is over tVoigtIndex_J.
            tExpEval.evaluate_expression( aCellOrdinal, tStress );

            // Sum the stress values.
            aCauchyStress(aCellOrdinal, tVoigtIndex_I) = 0.0;

            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
              aCauchyStress(aCellOrdinal, tVoigtIndex_I) += tStress(aCellOrdinal, tVoigtIndex_J);

              // The original stress equation.
              // aCauchyStress(aCellOrdinal, tVoigtIndex_I) +=
              //        ((aSmallStrain(aCellOrdinal, tVoigtIndex_J) - tReferenceStrain(tVoigtIndex_J)) +
              //         (aVelGrad(aCellOrdinal, tVoigtIndex_J) *  tRayleighB)) *
              //        tCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
      } );

      // Clear the temporary storage used in the expression
      // otherwise there will be memory leaks.
      tExpEval.clear_storage();
    }
};
// class HyperbolicLinearStressExpression

}// namespace Hyperbolic

}// namespace Plato

#endif

#ifdef PLATOANALYZE_1D
  PLATO_HYPERBOLIC_EXPL_DEC2(Plato::Hyperbolic::HyperbolicLinearStressExpression, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
  PLATO_HYPERBOLIC_EXPL_DEC2(Plato::Hyperbolic::HyperbolicLinearStressExpression, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
  PLATO_HYPERBOLIC_EXPL_DEC2(Plato::Hyperbolic::HyperbolicLinearStressExpression, Plato::SimplexMechanics, 3)
#endif
