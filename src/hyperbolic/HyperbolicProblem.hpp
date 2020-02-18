#ifndef PLATO_HYPERBOLIC_PROBLEM_HPP
#define PLATO_HYPERBOLIC_PROBLEM_HPP

#include "SimplexMechanics.hpp"
#include "ScalarFunctionBase.hpp"
#include "PlatoAbstractProblem.hpp"
#include "hyperbolic/VectorFunctionHyperbolic.hpp"
#include "hyperbolic/Newmark.hpp"
#include "hyperbolic/HyperbolicMechanics.hpp"

namespace Plato
{
    template<typename SimplexPhysics>
    class HyperbolicProblem: public Plato::AbstractProblem
    {
      private:
        static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::mNumSpatialDims;
        static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;


        Plato::VectorFunctionHyperbolic<SimplexPhysics> mEqualityConstraint;

        Plato::NewmarkIntegrator<SimplexPhysics>     mNewmarkIntegrator;

        Plato::OrdinalType mNumSteps;
        Plato::Scalar      mTimeStep;

        std::shared_ptr<const Plato::ScalarFunctionBase> mConstraint;
        std::shared_ptr<const Plato::ScalarFunctionBase> mObjective;

        Plato::ScalarVector      mResidual;
        Plato::ScalarVector      mResidualV;
        Plato::ScalarVector      mResidualA;

        Plato::ScalarMultiVector mAdjoints;
        Plato::ScalarMultiVector mDisplacement;
        Plato::ScalarMultiVector mVelocity;
        Plato::ScalarMultiVector mAcceleration;

        Teuchos::RCP<Plato::CrsMatrixType> mJacobian;
        Teuchos::RCP<Plato::CrsMatrixType> mJacobianV;
        Teuchos::RCP<Plato::CrsMatrixType> mJacobianA;

        Plato::LocalOrdinalVector mVelocityBcDofs;
        Plato::ScalarVector mVelocityBcValues;

      public:
        /******************************************************************************/
        HyperbolicProblem(
          Omega_h::Mesh& aMesh,
          Omega_h::MeshSets& aMeshSets,
          Teuchos::ParameterList& aParamList) :
            mEqualityConstraint   (aMesh, aMeshSets, mDataMap, aParamList, 
                                   aParamList.get<std::string>("PDE Constraint")),
            mNewmarkIntegrator    (aParamList.sublist("Time Integration")),
            mNumSteps     (aParamList.sublist("Time Integration").get<int>("Number Time Steps")),
            mTimeStep     (aParamList.sublist("Time Integration").get<Plato::Scalar>("Time Step")),
            mConstraint   (nullptr),
            mObjective    (nullptr),
            mResidual     ("MyResidual", mEqualityConstraint.size()),
            mDisplacement ("Displacement", mNumSteps, mEqualityConstraint.size()),
            mVelocity     ("Velocity",     mNumSteps, mEqualityConstraint.size()),
            mAcceleration ("Acceleration", mNumSteps, mEqualityConstraint.size()),
            mJacobian(Teuchos::null),
            mJacobianV(Teuchos::null),
            mJacobianA(Teuchos::null)
        /******************************************************************************/
        {
            // parse constraints
            //
            Plato::EssentialBCs<SimplexPhysics>
                tVelocityBoundaryConditions(aParamList.sublist("Velocity Boundary Conditions",false));
            tVelocityBoundaryConditions.get(aMeshSets, mVelocityBcDofs, mVelocityBcValues);
        }
        /******************************************************************************//**
         * @brief Return number of degrees of freedom in solution.
         * @return Number of degrees of freedom
        **********************************************************************************/
        Plato::OrdinalType getNumSolutionDofs()
        {
            return SimplexPhysics::mNumDofsPerNode;
        }
        /******************************************************************************/
        Plato::ScalarMultiVector getState()
        /******************************************************************************/
        {
            return mDisplacement;
        }

        /******************************************************************************/
        Plato::ScalarMultiVector getAdjoint()
        /******************************************************************************/
        {
            return mAdjoints;
        }
        /******************************************************************************/
        void setState(const Plato::ScalarMultiVector & aStates)
        /******************************************************************************/
        {
            assert(aStates.extent(0) == mDisplacement.extent(0));
            assert(aStates.extent(1) == mDisplacement.extent(1));
            Kokkos::deep_copy(mDisplacement, aStates);
        }
        void applyConstraints(
          const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
          const Plato::ScalarVector & aVector){}
  
        /******************************************************************************/
        void applyVelocityConstraints(
          const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
          const Plato::ScalarVector & aVector,
          Plato::Scalar aScale
        )
        /******************************************************************************/
        {
            if(mJacobian->isBlockMatrix())
            {
                Plato::applyBlockConstraints<mNumDofsPerNode>(aMatrix, aVector, mVelocityBcDofs, mVelocityBcValues, aScale);
            }
            else
            {
                Plato::applyConstraints<mNumDofsPerNode>(aMatrix, aVector, mVelocityBcDofs, mVelocityBcValues, aScale);
            }
        }
        void applyBoundaryLoads(const Plato::ScalarVector & aForce){}
        /******************************************************************************//**
         * @brief Update physics-based parameters within optimization iterations
         * @param [in] aState 2D container of state variables
         * @param [in] aControl 1D container of control variables
        **********************************************************************************/
        void updateProblem(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
        { return; }
        /******************************************************************************/
        Plato::ScalarMultiVector solution(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            for(Plato::OrdinalType tStepIndex = 1; tStepIndex < mNumSteps; tStepIndex++) {
              Plato::ScalarVector tDisplacementPrev = Kokkos::subview(mDisplacement, tStepIndex-1, Kokkos::ALL());
              Plato::ScalarVector tVelocityPrev     = Kokkos::subview(mVelocity,     tStepIndex-1, Kokkos::ALL());
              Plato::ScalarVector tAccelerationPrev = Kokkos::subview(mAcceleration, tStepIndex-1, Kokkos::ALL());

              Plato::ScalarVector tDisplacement = Kokkos::subview(mDisplacement, tStepIndex, Kokkos::ALL());
              Plato::ScalarVector tVelocity     = Kokkos::subview(mVelocity,     tStepIndex, Kokkos::ALL());
              Plato::ScalarVector tAcceleration = Kokkos::subview(mAcceleration, tStepIndex, Kokkos::ALL());


              // -R_{u}
              mResidual  = mEqualityConstraint.value(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep);
              Plato::scale(-1.0, mResidual);

              // R_{v}
              mResidualV = mNewmarkIntegrator.v_value(tDisplacement, tDisplacementPrev,
                                                      tVelocity,     tVelocityPrev,
                                                      tAcceleration, tAccelerationPrev, mTimeStep);

              // R_{u,v^N}
              mJacobianV = mEqualityConstraint.gradient_v(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep);

              // -R_{u} += R_{u,v^N} R_{v}
              Plato::MatrixTimesVectorPlusVector(mJacobianV, mResidualV, mResidual);

              // R_{a}
              mResidualA = mNewmarkIntegrator.a_value(tDisplacement, tDisplacementPrev,
                                                      tVelocity,     tVelocityPrev,
                                                      tAcceleration, tAccelerationPrev, mTimeStep);

              // R_{u,a^N}
              mJacobianA = mEqualityConstraint.gradient_a(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep);

              // -R_{u} += R_{u,a^N} R_{a}
              Plato::MatrixTimesVectorPlusVector(mJacobianA, mResidualA, mResidual);

              // R_{u,u^N}
              mJacobian  = mEqualityConstraint.gradient_u(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep);

              // R_{v,v}^{-1}
              auto tR_vv = mNewmarkIntegrator.v_grad_v_inverse(mTimeStep);

              // R_{u,u^N} += R_{v,v}^{-1} R_{u,v^N}
              Plato::axpy(tR_vv, mJacobianV->entries(), mJacobian->entries());

              // R_{a,a}^{-1}
              auto tR_aa = mNewmarkIntegrator.a_grad_a_inverse(mTimeStep);

              // R_{u,u^N} += R_{a,a}^{-1} R_{u,a^N}
              Plato::axpy(tR_aa, mJacobianA->entries(), mJacobian->entries());

              this->applyVelocityConstraints(mJacobian, mResidual, mTimeStep);

              Plato::ScalarVector tDeltaD("increment", tDisplacement.extent(0));
              Plato::fill(static_cast<Plato::Scalar>(0.0), tDeltaD);

              // compute displacement increment:
              Plato::Solve::Consistent<SimplexPhysics::mNumDofsPerNode>(mJacobian, tDeltaD, mResidual);

              // compute and add velocity increment: \Delta v = - ( R_{v} - R_{v,v}^{-1} \Delta u )
              Plato::axpy(-tR_vv, tDeltaD, mResidualV);
              // a_{k+1} = a_{k} + \Delta a
              Plato::axpy(-1.0, mResidualV, tVelocity);

              // compute and add acceleration increment: \Delta a = - ( R_{a} - R_{a,a}^{-1} \Delta u )
              Plato::axpy(-tR_aa, tDeltaD, mResidualA);
              // a_{k+1} = a_{k} + \Delta a
              Plato::axpy(-1.0, mResidualA, tAcceleration);

              // add displacement increment
              Plato::axpy(1.0, tDeltaD, tDisplacement);

              // evaluate at new state
              mResidual  = mEqualityConstraint.value(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep);
            }
            return mDisplacement;
        }
        /******************************************************************************/
        Plato::Scalar constraintValue(
          const Plato::ScalarVector & aControl,
          const Plato::ScalarMultiVector & aStates
        )
        /******************************************************************************/
        {
            return 0;
        }
        /******************************************************************************/
        Plato::Scalar constraintValue(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return 0;
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradient(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("constraint gradient", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradient(
          const Plato::ScalarVector & aControl,
          const Plato::ScalarMultiVector & aStates
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("constraint gradient", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradientX(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("constraint gradientX", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradientX(
          const Plato::ScalarVector & aControl,
          const Plato::ScalarMultiVector & aStates
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("constraint gradientX", 0);
        }
        /******************************************************************************/
        Plato::Scalar objectiveValue(
          const Plato::ScalarVector & aControl,
          const Plato::ScalarMultiVector & aStates
        )
        /******************************************************************************/
        {
            return 0;
        }
        /******************************************************************************/
        Plato::Scalar objectiveValue(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return 0;
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradient(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("objective gradient", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradient(
          const Plato::ScalarVector & aControl,
          const Plato::ScalarMultiVector & aStates
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("objective gradient", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradientX(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("objective gradientX", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradientX(
          const Plato::ScalarVector & aControl,
          const Plato::ScalarMultiVector & aStates
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("objective gradientX", 0);
        }
    };
}

#endif
