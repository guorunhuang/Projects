package assignment1.problem3;

public abstract class GroupFiler extends TaxFiler {
  protected int numDependents;
  protected int numMinorChildren;
  protected double childcareExpenses;
  protected double dependentCareExpenses;

  private static final double SAVINGS_COEFF = 0.65;
  private static final double SAVINGS_CAP = 17500.0;
  private static final double EARNINGS_LIMIT_CHILDCARE = 200000.0;
  private static final double EXPENSE_LIMIT_CHILDCARE = 5000.0;
  private static final double CHILDCARE_DEDUCTION_AMT = 1250.0;
  private static final double INCOME_THRESHOLD = 90000.0;
  private static final double LOW_RATE = 0.145;
  private static final double HIGH_RATE = 0.185;

  public GroupFiler(String id, ContactInfo info, double earnings, double taxPaid,
      double m, double p, double s, double r, double h, double c,
      int dependents, int minors, double childcare, double depCare) {
    super(id, info, earnings, taxPaid, m, p, s, r, h, c);
    this.numDependents = dependents;
    this.numMinorChildren = minors;
    this.childcareExpenses = childcare;
    this.dependentCareExpenses = depCare;
  }

  @Override
  protected double getRetirementHealthDeduction() {
    double deduction = (this.retirementContribution + this.healthSavingsContribution) * SAVINGS_COEFF;
    return Math.min(deduction, SAVINGS_CAP);
  }

  @Override
  protected double getChildcareDeduction() {
    if (this.lastYearEarnings < EARNINGS_LIMIT_CHILDCARE &&
        this.childcareExpenses > EXPENSE_LIMIT_CHILDCARE) {
      return CHILDCARE_DEDUCTION_AMT;
    }
    return 0.0;
  }

  @Override
  protected double applyTaxRate(double taxableIncome) {
    double rate = (taxableIncome < INCOME_THRESHOLD) ? LOW_RATE : HIGH_RATE;
    return taxableIncome * rate;
  }
}
