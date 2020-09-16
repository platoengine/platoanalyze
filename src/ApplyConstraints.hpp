#ifndef APPLY_CONSTRAINTS_HPP
#define APPLY_CONSTRAINTS_HPP

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Constrain all Dofs of given nodes
 * \param [in] aMatrix Matrix to be constrained
 * \param [in] aRhs    Vector to be constrained
 * \param [in] aNodes  List of node ids to be constrained
**********************************************************************************/
template<int NumDofPerNode>
void
applyBlockConstraints(
    Teuchos::RCP<Plato::CrsMatrixType>  aMatrix,
    Plato::ScalarVector                 aRhs,
    Plato::LocalOrdinalVector           aNodes
)
/******************************************************************************/
{
    Plato::OrdinalType tNumNodes = aNodes.size();

    Plato::LocalOrdinalVector tDofs("dof ids", tNumNodes * NumDofPerNode);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        for(Plato::OrdinalType tDof=0; tDof<NumDofPerNode; tDof++)
        {
            tDofs(aNodeOrdinal*NumDofPerNode + tDof) = aNodes(aNodeOrdinal)*NumDofPerNode+tDof;
        }
    }, "dof ids");

    Plato::ScalarVector tVals("values", tDofs.extent(0));
    applyBlockConstraints<NumDofPerNode>(aMatrix, aRhs, tDofs, tVals, 1.0);
}
  
/******************************************************************************/
template<int NumDofPerNode>
void
applyBlockConstraints(
    Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
    Plato::ScalarVector                aRhs,
    Plato::LocalOrdinalVector          aDirichletDofs,
    Plato::ScalarVector                aDirichletValues,
    Plato::Scalar                      aScale=1.0
  )
/******************************************************************************/
{
  
  /*
   Because both MAGMA and ViennaCL apparently assume symmetry even though it's technically
   not required for CG (and they do so in a way that breaks the solve badly), we do make the
   effort here to maintain symmetry while imposing BCs.
   */
    Plato::OrdinalType tNumBCs = aDirichletDofs.size();
    auto tRowMap        = aMatrix->rowMap();
    auto tColumnIndices = aMatrix->columnIndices();
    ScalarVector tMatrixEntries = aMatrix->entries();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumBCs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aBcOrdinal)
    {
        OrdinalType tRowDofOrdinal = aDirichletDofs[aBcOrdinal];
        Scalar tValue = aScale*aDirichletValues[aBcOrdinal];
        auto tRowNodeOrdinal = tRowDofOrdinal / NumDofPerNode;
        auto tLocalRowDofOrdinal  = tRowDofOrdinal % NumDofPerNode;
        OrdinalType tRowStart = tRowMap(tRowNodeOrdinal  );
        OrdinalType tRowEnd   = tRowMap(tRowNodeOrdinal+1);
        for (OrdinalType tColumnNodeOffset=tRowStart; tColumnNodeOffset<tRowEnd; tColumnNodeOffset++)
        {
            for (OrdinalType tLocalColumnDofOrdinal=0; tLocalColumnDofOrdinal<NumDofPerNode; tLocalColumnDofOrdinal++)
            {
                OrdinalType tColumnNodeOrdinal = tColumnIndices(tColumnNodeOffset);
                auto tEntryOrdinal = NumDofPerNode*NumDofPerNode*tColumnNodeOffset
                        + NumDofPerNode*tLocalRowDofOrdinal + tLocalColumnDofOrdinal;
                auto tColumnDofOrdinal = NumDofPerNode*tColumnNodeOrdinal+tLocalColumnDofOrdinal;
                if (tColumnDofOrdinal == tRowDofOrdinal) // diagonal
                {
                    tMatrixEntries(tEntryOrdinal) = 1.0;
                }
                else
                {
                    // correct the rhs to account for the fact that we'll be zeroing out (col,row) as well
                    // to maintain symmetry
                    Kokkos::atomic_add(&aRhs(tColumnDofOrdinal), -tMatrixEntries(tEntryOrdinal)*tValue);
                    tMatrixEntries(tEntryOrdinal) = 0.0;
                    OrdinalType tColRowStart = tRowMap(tColumnNodeOrdinal  );
                    OrdinalType tColRowEnd   = tRowMap(tColumnNodeOrdinal+1);
                    for (OrdinalType tColRowNodeOffset=tColRowStart; tColRowNodeOffset<tColRowEnd; tColRowNodeOffset++)
                    {
                        OrdinalType tColRowNodeOrdinal = tColumnIndices(tColRowNodeOffset);
                        auto tColRowEntryOrdinal = NumDofPerNode*NumDofPerNode*tColRowNodeOffset
                                +NumDofPerNode*tLocalColumnDofOrdinal+tLocalRowDofOrdinal;
                        auto tColRowDofOrdinal = NumDofPerNode*tColRowNodeOrdinal+tLocalRowDofOrdinal;
                        if (tColRowDofOrdinal == tRowDofOrdinal)
                        {
                            // this is the (col, row) entry -- clear it, too
                            tMatrixEntries(tColRowEntryOrdinal) = 0.0;
                        }
                    }
                }
            }
        }
    },"Dirichlet BC imposition - First loop");
  
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumBCs), LAMBDA_EXPRESSION(int bcOrdinal){
        OrdinalType tDofOrdinal = aDirichletDofs[bcOrdinal];
        Scalar tValue = aScale*aDirichletValues[bcOrdinal];
        aRhs(tDofOrdinal) = tValue;
    },"Dirichlet BC imposition - Second loop");

}

/******************************************************************************/
template<int NumDofPerNode> void
applyConstraints(
    Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
    Plato::ScalarVector                aRhs,
    Plato::LocalOrdinalVector          aNodes
)
/******************************************************************************/
{
    Plato::OrdinalType tNumNodes = aNodes.size();

    Plato::LocalOrdinalVector tDofs("dof ids", tNumNodes * NumDofPerNode);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        for(Plato::OrdinalType tDof=0; tDof<NumDofPerNode; tDof++)
        {
            tDofs(aNodeOrdinal*NumDofPerNode + tDof) = aNodes(aNodeOrdinal)*NumDofPerNode+tDof;
        }
    }, "dof ids");

    Plato::ScalarVector tVals("values", tDofs.extent(0));
    applyConstraints<NumDofPerNode>(aMatrix, aRhs, tDofs, tVals, 1.0);
}
  
/******************************************************************************/
template<int NumDofPerNode> void
applyConstraints(
    Teuchos::RCP<Plato::CrsMatrixType> matrix,
    Plato::ScalarVector                rhs,
    Plato::LocalOrdinalVector          bcDofs,
    Plato::ScalarVector                bcValues,
    Plato::Scalar                      aScale=1.0
)
/******************************************************************************/
{
  
  /*
   Because both MAGMA and ViennaCL apparently assume symmetry even though it's technically
   not required for CG (and they do so in a way that breaks the solve badly), we do make the
   effort here to maintain symmetry while imposing BCs.
   */
  int numBCs = bcDofs.size();
  auto rowMap        = matrix->rowMap();
  auto columnIndices = matrix->columnIndices();
  ScalarVector matrixEntries = matrix->entries();
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numBCs), LAMBDA_EXPRESSION(int bcOrdinal)
  {
    OrdinalType nodeNumber = bcDofs[bcOrdinal];
    Scalar value = aScale*bcValues[bcOrdinal];
    OrdinalType rowStart = rowMap(nodeNumber  );
    OrdinalType rowEnd   = rowMap(nodeNumber+1);
    for (OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
    {
      OrdinalType column = columnIndices(entryOrdinal);
      if (column == nodeNumber) // diagonal
      {
        matrixEntries(entryOrdinal) = 1.0;
      }
      else
      {
        // correct the rhs to account for the fact that we'll be zeroing out (col,row) as well
        // to maintain symmetry
        Kokkos::atomic_add(&rhs(column), -matrixEntries(entryOrdinal)*value);
        matrixEntries(entryOrdinal) = 0.0;
        OrdinalType colRowStart = rowMap(column  );
        OrdinalType colRowEnd   = rowMap(column+1);
        for (OrdinalType colRowEntryOrdinal=colRowStart; colRowEntryOrdinal<colRowEnd; colRowEntryOrdinal++)
        {
          OrdinalType colRowColumn = columnIndices(colRowEntryOrdinal);
          if (colRowColumn == nodeNumber)
          {
            // this is the (col, row) entry -- clear it, too
            matrixEntries(colRowEntryOrdinal) = 0.0;
          }
        }
      }
    }
  },"BC imposition");
  
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numBCs), LAMBDA_EXPRESSION(int bcOrdinal)
  {
    OrdinalType nodeNumber = bcDofs[bcOrdinal];
    Scalar value = aScale*bcValues[bcOrdinal];
    rhs(nodeNumber) = value;
  },"BC imposition");

}

} // namespace Plato


#endif
