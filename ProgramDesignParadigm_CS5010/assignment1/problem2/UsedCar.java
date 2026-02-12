package assignment1.problem2;

/**
 * Represents a used car with history.
 */
public class UsedCar extends Vehicle {
  private Integer mileage;
  private Integer previousOwners;
  private Integer accidents;

  public UsedCar(String id, Integer year, MakeModel mm, double msrp,
      Integer mileage, Integer owners, Integer accidents) {
    super(id, year, mm, msrp);
    this.mileage = mileage;
    this.previousOwners = owners;
    this.accidents = accidents;
  }

  public Integer getMileage() { return this.mileage; }
  public Integer getPreviousOwners() { return this.previousOwners; }
  public Integer getAccidents() { return this.accidents; }
}