package assignment1.problem3;

/**
 * Represents married individuals filing separately.
 */
public class MarriedSeparately extends GroupFiler {
  public MarriedSeparately(String taxId, ContactInfo contactInfo, double earnings,
      double taxPaid, double m, double p, double s, double r,
      double h, double c, int dependents, int minors,
      double childcare, double depCare) {
    super(taxId, contactInfo, earnings, taxPaid, m, p, s, r, h, c,
        dependents, minors, childcare, depCare);
  }
}