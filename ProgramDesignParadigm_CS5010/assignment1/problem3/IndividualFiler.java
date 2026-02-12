package assignment1.problem3;

/**
 * Abstract class representing an individual tax filer.
 */
public abstract class IndividualFiler extends TaxFiler {
  private static final double SAVINGS_COEFFICIENT = 0.7;
  private static final double TAXABLE_INCOME_THRESHOLD = 55000.0;
  private static final double LOW_TAX_RATE = 0.15;
  private static final double HIGH_TAX_RATE = 0.19;

  public IndividualFiler(String taxId, ContactInfo contactInfo, double earnings,
      double taxPaid, double mortgage, double property,
      double student, double retirement, double hsa, double charity) {
    super(taxId, contactInfo, earnings, taxPaid, mortgage, property,
        student, retirement, hsa, charity);
  }

  @Override
  protected double getRetirementHealthDeduction() {
    return (this.retirementContribution + this.healthSavingsContribution) * SAVINGS_COEFFICIENT;
  }

  @Override
  protected double getChildcareDeduction() {
    return 0.0; // Individuals do not have childcare deduction rules in this system
  }

  @Override
  protected double applyTaxRate(double taxableIncome) {
    double rate = (taxableIncome < TAXABLE_INCOME_THRESHOLD) ? LOW_TAX_RATE : HIGH_TAX_RATE;
    return taxableIncome * rate;
  }
}
