package assignment1.problem3;

/**
 * Represents a head of household tax filer.
 */
public class HeadOfHousehold extends GroupFiler {
  public HeadOfHousehold(String taxId, ContactInfo contactInfo, double earnings,
      double taxPaid, double m, double p, double s, double r,
      double h, double c, int dependents, int minors,
      double childcare, double depCare) {
    super(taxId, contactInfo, earnings, taxPaid, m, p, s, r, h, c,
        dependents, minors, childcare, depCare);
  }
}