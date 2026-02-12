package assignment1.problem3;

/**
 * Abstract class representing a general tax filer.
 */
public abstract class TaxFiler {
    protected String taxId;
    protected ContactInfo contactInfo;
    protected double lastYearEarnings;
    protected double incomeTaxPaid;
    protected double mortgageInterest;
    protected double propertyTax;
    protected double studentLoan;
    protected double retirementContribution;
    protected double healthSavingsContribution;
    protected double charitableDonation;

    // Constants to avoid Magic Numbers
    protected static final double EARNINGS_LIMIT_MORTGAGE = 250000.0;
    protected static final double EXPENSE_LIMIT_MORTGAGE = 12500.0;
    protected static final double MORTGAGE_DEDUCTION = 2500.0;

    public TaxFiler(String taxId, ContactInfo contactInfo, double earnings, double taxPaid,
        double mortgage, double property, double student, double retirement,
        double hsa, double charity) {
        this.taxId = taxId;
        this.contactInfo = contactInfo;
        this.lastYearEarnings = earnings;
        this.incomeTaxPaid = taxPaid;
        this.mortgageInterest = mortgage;
        this.propertyTax = property;
        this.studentLoan = student;
        this.retirementContribution = retirement;
        this.healthSavingsContribution = hsa;
        this.charitableDonation = charity;
    }

    /**
     * Calculates the tax amount based on the specific rules.
     * @return total tax amount
     */
    public double calculateTaxes() {
        double taxableIncome = this.lastYearEarnings - this.incomeTaxPaid;

        // 1. Retirement and Health Savings Deduction
        taxableIncome -= this.getRetirementHealthDeduction();
        if (taxableIncome < 0) {
            taxableIncome = 0;
        }

        // 2. Mortgage and Property Deduction
        if (this.lastYearEarnings < EARNINGS_LIMIT_MORTGAGE &&
            (this.mortgageInterest + this.propertyTax) > EXPENSE_LIMIT_MORTGAGE) {
            taxableIncome -= MORTGAGE_DEDUCTION;
        }

        // 3. Childcare Deduction (Only for Group Filers)
        taxableIncome -= this.getChildcareDeduction();

        if (taxableIncome < 0) taxableIncome = 0;

        // 4. Final Tax Calculation
        return this.applyTaxRate(taxableIncome);
    }

    protected abstract double getRetirementHealthDeduction();
    protected abstract double getChildcareDeduction();
    protected abstract double applyTaxRate(double taxableIncome);
}