package assignment1.problem2;

/**
 * Abstract class representing a general vehicle.
 */
public abstract class Vehicle {
  private String id;
  private Integer manufacturingYear;
  private MakeModel makeModel;
  private double msrp;

  /**
   * Constructor for Vehicle.
   * @param id unique identifier
   * @param year manufacturing year
   * @param makeModel make and model object
   * @param msrp suggested retail price
   */
  public Vehicle(String id, Integer year, MakeModel makeModel, double msrp) {
    this.id = id;
    this.manufacturingYear = year;
    this.makeModel = makeModel;
    this.msrp = msrp;
  }

  public String getId() { return this.id; }
  public Integer getManufacturingYear() { return this.manufacturingYear; }
  public MakeModel getMakeModel() { return this.makeModel; }
  public double getMsrp() { return this.msrp; }
}